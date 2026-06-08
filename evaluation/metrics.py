"""
RAG evaluation metrics.

hit_rate  — fraction of queries where the relevant doc appeared in top-k results
mrr       — Mean Reciprocal Rank
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def hit_rate(relevance_total: list[list[bool]]) -> float:
    """Fraction of queries that had at least one relevant result."""
    if not relevance_total:
        return 0.0
    hits = sum(1 for line in relevance_total if True in line)
    return hits / len(relevance_total)


def mrr(relevance_total: list[list[bool]]) -> float:
    """Mean Reciprocal Rank."""
    if not relevance_total:
        return 0.0
    score = 0.0
    for line in relevance_total:
        for rank, relevant in enumerate(line):
            if relevant:
                score += 1 / (rank + 1)
                break
    return score / len(relevance_total)


def evaluate_retriever(retriever, ground_truth_path: str) -> dict:
    """
    Run hit_rate and MRR against a ground truth JSON file.

    Ground truth format:
        [{"query": "...", "expected_doc_id": "..."}, ...]
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.warning(f"Ground truth file not found: {ground_truth_path}")
        return {"hit_rate": None, "mrr": None}

    with open(gt_path) as f:
        ground_truth = json.load(f)

    relevance_total = []
    for item in ground_truth:
        results = retriever.invoke(item["query"])
        relevance = [
            item["expected_doc_id"] in (doc.metadata.get("source", "") + doc.page_content)
            for doc in results
        ]
        relevance_total.append(relevance)

    metrics = {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
        "n_queries": len(ground_truth),
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics
