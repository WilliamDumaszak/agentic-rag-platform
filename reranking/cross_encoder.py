"""
Cross-encoder reranker using sentence-transformers.

Why cross-encoders?
  Bi-encoders (like BGE) encode query and document independently → fast but less accurate.
  Cross-encoders process the (query, document) pair together → slower but significantly
  more accurate relevance scoring. Used as a second stage after fast retrieval.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking (110k queries)
  - 22M parameters — fast on CPU
  - Returns a raw logit score (higher = more relevant)
"""

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_MODEL_NAME: str = CONFIG.get("reranking", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_TOP_K: int = CONFIG.get("reranking", {}).get("top_k", 3)
_ENABLED: bool = CONFIG.get("reranking", {}).get("enabled", True)

_reranker: Any = None  # lazy-loaded CrossEncoder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        logger.info(f"Loading cross-encoder: {_MODEL_NAME}")
        _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


def rerank(query: str, documents: list) -> list:
    """
    Rerank a list of LangChain Documents using a cross-encoder.

    Returns top-k documents sorted by relevance (highest score first).
    Falls back to original order if reranking fails or is disabled.
    """
    if not _ENABLED or not documents:
        return documents

    try:
        model = _get_reranker()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs).tolist()

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        top_k = min(_TOP_K, len(ranked))

        logger.info(
            f"Reranked {len(documents)} docs → top {top_k}, "
            f"best_score={ranked[0][0]:.4f}, worst_score={ranked[-1][0]:.4f}"
        )
        return [doc for _, doc in ranked[:top_k]]

    except Exception as exc:
        logger.warning(f"Reranking failed, returning original order: {exc}")
        return documents[:_TOP_K]
