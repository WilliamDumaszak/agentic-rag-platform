"""
Hybrid retriever — dual-provider: local (BM25 + vector + RRF) or Azure AI Search native hybrid.

Why hybrid retrieval?
  BM25 (keyword) excels at exact matches: product codes, names, acronyms.
  Vector search (semantic) excels at meaning: synonyms, paraphrases, context.
  Neither alone is best. Hybrid combines both via Reciprocal Rank Fusion (RRF),
  which consistently outperforms either method individually on BEIR benchmarks.

RRF formula: score(d) = Σ 1 / (k + rank_i(d))  where k=60 (standard constant)

Local provider: runs both retrievers in Python, fuses manually.
Azure provider: Azure AI Search does this natively in a single API call — even faster,
                and supports semantic re-ranking on top of hybrid.
"""

import logging
from typing import Any

import yaml
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_PROVIDER: str = CONFIG.get("retrieval", {}).get("provider", "local")
_TOP_K: int = CONFIG.get("retrieval", {}).get("top_k", 5)
_RRF_K: int = CONFIG.get("retrieval", {}).get("rrf_k", 60)


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Fuse multiple ranked lists of documents using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each sub-list is a ranked result from one retriever.
        k: RRF constant (default 60, from original paper).

    Returns:
        Single merged list sorted by fused RRF score, deduplicated.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            # Use content hash as deduplication key
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    logger.info(
        f"RRF fusion: {sum(len(r) for r in ranked_lists)} total → "
        f"{len(sorted_keys)} unique docs"
    )
    return [doc_map[k] for k in sorted_keys]


# ── Local: BM25 + vector ──────────────────────────────────────────────────────

def _retrieve_local(query: str) -> list[Document]:
    """
    Run BM25 and vector retrieval in parallel, fuse with RRF.

    BM25Retriever: keyword-based, uses TF-IDF style scoring over stored docs.
    Chroma vector: dense retrieval using BGE embeddings.
    """
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever

    from agent.nodes import _build_embeddings

    embeddings = _build_embeddings()

    # Vector retrieval
    db = Chroma(
        persist_directory=CONFIG["vectordb"]["persist_directory"],
        collection_name=CONFIG["vectordb"]["collection_name"],
        embedding_function=embeddings,
    )
    vector_results = db.similarity_search(query, k=_TOP_K)

    # BM25 retrieval over the same corpus
    all_docs = db.get()
    if not all_docs or not all_docs.get("documents"):
        logger.warning("ChromaDB is empty, returning vector results only.")
        return vector_results[:_TOP_K]

    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        if text
    ]

    if not bm25_docs:
        return vector_results[:_TOP_K]

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = _TOP_K
    bm25_results = bm25_retriever.invoke(query)

    fused = _reciprocal_rank_fusion([vector_results, bm25_results], k=_RRF_K)
    return fused[:_TOP_K]


# ── Azure AI Search: native hybrid ───────────────────────────────────────────

def _retrieve_azure(query: str) -> list[Document]:
    """
    Run Azure AI Search hybrid retrieval (keyword + vector in one API call).

    Azure AI Search combines BM25 and vector internally, then applies
    semantic re-ranking (L2 model) on top. This is equivalent to our
    local RRF but done server-side — lower latency, no extra calls.

    query_type="semantic" enables the semantic ranker.
    semantic_configuration_name must be configured in the Azure portal.
    """
    import os

    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery
    from azure.core.credentials import AzureKeyCredential

    from agent.nodes import _build_embeddings

    endpoint = os.getenv(
        "AZURE_SEARCH_ENDPOINT",
        CONFIG["vectordb"].get("azure_search_endpoint", ""),
    )
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    index_name = CONFIG["vectordb"].get("azure_search_index", "rag-documents")
    semantic_config = CONFIG.get("retrieval", {}).get(
        "azure_semantic_config", "default"
    )

    client = SearchClient(endpoint, index_name, AzureKeyCredential(api_key))

    # Embed query for the vector part
    embeddings = _build_embeddings()
    query_vector = embeddings.embed_query(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=_TOP_K,
        fields="content_vector",
    )

    results = client.search(
        search_text=query,             # BM25 keyword part
        vector_queries=[vector_query], # vector part
        query_type="semantic",         # semantic re-ranking on top
        semantic_configuration_name=semantic_config,
        top=_TOP_K,
    )

    documents = []
    for r in results:
        documents.append(Document(
            page_content=r.get("content", ""),
            metadata={
                "source": r.get("source", "unknown"),
                "score": r.get("@search.score", 0.0),
                "reranker_score": r.get("@search.reranker_score", None),
                "provider": "azure_search_hybrid",
            },
        ))

    logger.info(f"[azure hybrid] Retrieved {len(documents)} documents for query.")
    return documents


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_retrieve(query: str, provider: str | None = None) -> list[Document]:
    """
    Retrieve documents using hybrid search (BM25 + vector + RRF).

    Args:
        query:    User query string.
        provider: "local" | "azure". Defaults to config value.

    Returns:
        Top-k documents ranked by fused relevance score.
    """
    resolved_provider = provider or _PROVIDER
    logger.info(f"Hybrid retrieve via provider={resolved_provider}, query='{query[:60]}'")

    try:
        if resolved_provider == "azure":
            return _retrieve_azure(query)
        return _retrieve_local(query)
    except Exception as exc:
        logger.error(f"Hybrid retrieval failed: {exc}", exc_info=True)
        return []
