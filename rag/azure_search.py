"""
Azure AI Search helper — drop-in alternative to ChromaDB for cloud deployments.

Provides ingest and retrieval functions using Azure Cognitive Search REST API.
Used by agent/nodes.py when vectordb.provider == "azure_search" in config.yaml.
"""

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def _get_search_client():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", CONFIG["vectordb"].get("azure_search_endpoint", ""))
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    index_name = CONFIG["vectordb"]["azure_search_index"]

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )


def _get_index_client():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", CONFIG["vectordb"].get("azure_search_endpoint", ""))
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    return SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))


def _build_embeddings():
    from rag.ingestion import build_embedding_model

    return build_embedding_model()


def _get_embedding_dimensions() -> int:
    """Infer vector dimensions from the configured embedding model."""
    probe = _build_embeddings().embed_query("dimension probe")
    return len(probe)


def ensure_index() -> None:
    """
    Create the Azure AI Search index if it does not exist.

    Primary path creates a hybrid-compatible schema with:
      - content (text)
      - content_vector (vector)
      - semantic config for content field

    Fallback path creates a BM25-only schema when vector/semantic classes are
    unavailable in the installed SDK or service tier.
    """
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchableField,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    index_client = _get_index_client()
    index_name = CONFIG["vectordb"]["azure_search_index"]
    semantic_config = CONFIG.get("retrieval", {}).get("azure_semantic_config", "default")

    # Hybrid schema (text + vector + semantic config)
    try:
        dims = _get_embedding_dimensions()
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dims,
                vector_search_profile_name="default",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[
                VectorSearchProfile(
                    name="default",
                    algorithm_configuration_name="hnsw-config",
                )
            ],
        )

        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=semantic_config,
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="content")]
                    ),
                )
            ]
        )

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        index_client.create_or_update_index(index)
        logger.info(f"Azure AI Search hybrid index '{index_name}' ready.")
        return
    except Exception as exc:
        logger.warning(
            "Hybrid index creation failed, falling back to BM25-only schema: %s",
            exc,
        )

    # BM25-only fallback schema
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
    ]
    index = SearchIndex(name=index_name, fields=fields)
    index_client.create_or_update_index(index)
    logger.info(f"Azure AI Search index '{index_name}' ready.")


def ingest_documents(documents: list[dict]) -> int:
    """
    Upload documents to Azure AI Search.
    Each doc must have: id (str), content (str), source (str).
    Returns number of documents indexed.
    """
    client = _get_search_client()

    # Ensure index exists before first ingest.
    try:
        ensure_index()
    except Exception as exc:
        logger.warning(f"Index ensure skipped/failed: {exc}")

    # Add vectors for docs that don't provide content_vector yet.
    docs_to_upload: list[dict[str, Any]] = []
    missing_vector_docs = [d for d in documents if d.get("content") and not d.get("content_vector")]
    vectors_by_index: dict[int, list[float]] = {}

    if missing_vector_docs:
        try:
            embeddings = _build_embeddings()
            vectors = embeddings.embed_documents([d["content"] for d in missing_vector_docs])
            for idx, vec in enumerate(vectors):
                vectors_by_index[idx] = vec
        except Exception as exc:
            logger.warning(f"Vector embedding during ingest failed, uploading without vectors: {exc}")

    mv_i = 0
    for doc in documents:
        payload = dict(doc)
        if payload.get("content") and not payload.get("content_vector") and mv_i in vectors_by_index:
            payload["content_vector"] = vectors_by_index[mv_i]
            mv_i += 1
        elif payload.get("content") and not payload.get("content_vector"):
            mv_i += 1
        docs_to_upload.append(payload)

    batch = [{"@search.action": "upload", **doc} for doc in docs_to_upload]
    result = client.upload_documents(documents=batch)
    succeeded = sum(1 for r in result if r.succeeded)
    logger.info(f"Indexed {succeeded}/{len(batch)} documents to Azure AI Search.")
    return succeeded


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Full-text BM25 search. Returns list of source dicts."""
    client = _get_search_client()
    k = top_k or CONFIG["vectordb"].get("top_k", 5)
    results = client.search(search_text=query, top=k)
    return [dict(r) for r in results]
