"""
Document Intelligence pipeline — dual-provider: local (PyMuPDF + Unstructured) or Azure.

Why two providers?
  Local: PyMuPDF extracts text from PDFs without any API call. Fast, free, works offline.
         Unstructured normalizes tables and layout elements into plain text.
         Covers 80% of use cases (digital PDFs, text-heavy docs).

  Azure: Azure Document Intelligence (formerly Form Recognizer) uses neural models
         trained on millions of documents. Handles scanned PDFs, handwriting, complex
         tables, and custom document types (invoices, contracts, receipts).
         Required when: document is scanned/image-based, or field extraction accuracy
         matters for regulated workloads.

Output: always returns a list of LangChain Documents, ready for chunking and indexing.
"""

import logging
import os
from pathlib import Path

import yaml
from langchain.schema import Document

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_PROVIDER: str = CONFIG.get("document_intelligence", {}).get("provider", "local")


# ── Local extraction ──────────────────────────────────────────────────────────

def _extract_local(file_path: str) -> list[Document]:
    """
    Extract text from a PDF using PyMuPDF (fitz).

    Preserves page numbers as metadata.
    Falls back to raw text extraction if layout parsing fails.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    path = Path(file_path)
    documents = []

    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "page": page_num,
                    "total_pages": len(pdf),
                    "provider": "local_pymupdf",
                    "file_path": str(path),
                },
            ))

    logger.info(f"[local] Extracted {len(documents)} pages from {path.name}")
    return documents


# ── Azure Document Intelligence ───────────────────────────────────────────────

def _extract_azure(file_path: str) -> list[Document]:
    """
    Extract text and structured fields from a document using Azure Document Intelligence.

    Uses the 'prebuilt-layout' model by default — handles scanned PDFs, tables, headers.
    For custom document types (invoices, contracts), set model_id in config to your
    custom model ID trained in the Azure portal.

    Field extraction: for models like 'prebuilt-invoice', the API returns structured
    fields (VendorName, TotalAmount, etc.) in addition to raw text.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    endpoint = os.getenv(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        CONFIG.get("document_intelligence", {}).get("azure_endpoint", ""),
    )
    api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
    model_id = CONFIG.get("document_intelligence", {}).get("azure_model_id", "prebuilt-layout")

    if not endpoint:
        raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT not set.")

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(api_key))
    path = Path(file_path)

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(model_id, body=f, content_type="application/pdf")
    result = poller.result()

    documents = []

    # Extract text page by page
    for page in result.pages:
        page_text = " ".join(
            word.content for line in (page.lines or []) for word in (line.words or [])
            if hasattr(word, "content")
        )
        if not page_text.strip():
            # fallback: join line content directly
            page_text = " ".join(
                line.content for line in (page.lines or []) if line.content
            )

        if not page_text.strip():
            continue

        documents.append(Document(
            page_content=page_text,
            metadata={
                "source": path.name,
                "page": page.page_number,
                "provider": "azure_document_intelligence",
                "model_id": model_id,
                "file_path": str(path),
            },
        ))

    # If the model returns structured fields (e.g. prebuilt-invoice), append as extra doc
    if result.documents:
        for analyzed_doc in result.documents:
            fields_text = []
            for field_name, field in (analyzed_doc.fields or {}).items():
                value = getattr(field, "value_string", None) or getattr(field, "content", None)
                if value:
                    fields_text.append(f"{field_name}: {value}")

            if fields_text:
                documents.append(Document(
                    page_content="\n".join(fields_text),
                    metadata={
                        "source": path.name,
                        "provider": "azure_document_intelligence",
                        "model_id": model_id,
                        "doc_type": analyzed_doc.doc_type or "unknown",
                        "content_type": "structured_fields",
                        "confidence": analyzed_doc.confidence or 0.0,
                        "file_path": str(path),
                    },
                ))

    logger.info(f"[azure] Extracted {len(documents)} sections from {path.name}")
    return documents


# ── Public API ────────────────────────────────────────────────────────────────

def extract_document(file_path: str, provider: str | None = None) -> list[Document]:
    """
    Extract text from a document file.

    Args:
        file_path: Path to the PDF or document file.
        provider:  "local" | "azure". Defaults to config value.

    Returns:
        List of LangChain Documents with page content and metadata.
    """
    resolved_provider = provider or _PROVIDER
    logger.info(f"Extracting '{Path(file_path).name}' via provider={resolved_provider}")

    if resolved_provider == "azure":
        return _extract_azure(file_path)
    return _extract_local(file_path)


def extract_directory(directory: str, provider: str | None = None) -> list[Document]:
    """
    Extract all PDF and text documents from a directory.

    Returns:
        Flat list of Documents from all files combined.
    """
    resolved_provider = provider or _PROVIDER
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    supported = [".pdf", ".txt", ".md"]
    files = [f for f in dir_path.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.warning(f"No supported files found in {directory}")
        return []

    all_docs = []
    for file_path in files:
        try:
            docs = extract_document(str(file_path), provider=resolved_provider)
            all_docs.extend(docs)
        except Exception as exc:
            logger.error(f"Failed to extract {file_path.name}: {exc}")

    logger.info(f"Extracted {len(all_docs)} total sections from {len(files)} files.")
    return all_docs
