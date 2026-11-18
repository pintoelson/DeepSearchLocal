"""Information Retrieval (IR) ranking metrics for evaluating retrieval modes."""

from __future__ import annotations

import math
from typing import Any, Sequence


def calculate_hit_rate(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Hit Rate@K (1.0 if at least one relevant document is in top-K, else 0.0)."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(item in relevant_set for item in top_k) else 0.0


def calculate_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Recall@K (fraction of all relevant documents captured in top-K)."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    hits = len(top_k_set & relevant_set)
    return hits / len(relevant_set)


def calculate_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Precision@K (fraction of top-K retrieved documents that are relevant)."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k_set = set(retrieved_ids[:k])
    hits = len(top_k_set & relevant_set)
    return hits / float(k)


def calculate_mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | Sequence[str],
) -> float:
    """Compute Reciprocal Rank (1.0 / rank of first relevant item, or 0.0 if none retrieved)."""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    for rank_0, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (rank_0 + 1)
    return 0.0


def calculate_dcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_map: dict[str, float | int],
    k: int,
) -> float:
    """Compute Discounted Cumulative Gain at rank K using binary or graded relevance."""
    if k <= 0:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = float(relevance_map.get(doc_id, 0.0))
        if rel > 0.0:
            # Standard formulation: (2^rel - 1) / log2(rank + 1)
            rank = i + 1
            gain = (2.0**rel) - 1.0
            discount = math.log2(rank + 1)
            dcg += gain / discount
    return dcg


def calculate_ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_map: dict[str, float | int],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    if k <= 0:
        return 0.0
    dcg = calculate_dcg_at_k(retrieved_ids, relevance_map, k)

    # Compute Ideal DCG (IDCG) by sorting relevant values descending
    all_gains = sorted(relevance_map.values(), reverse=True)
    if not all_gains or all_gains[0] <= 0.0:
        return 0.0

    idcg = 0.0
    for i, rel in enumerate(all_gains[:k]):
        rel_f = float(rel)
        if rel_f > 0.0:
            rank = i + 1
            gain = (2.0**rel_f) - 1.0
            discount = math.log2(rank + 1)
            idcg += gain / discount

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_result(
    retrieved_ids: Sequence[str],
    ground_truth_relevance: dict[str, float | int],
    k_values: Sequence[int] = (1, 3, 5, 10, 15),
) -> dict[str, float]:
    """Compute full set of IR metrics for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        ground_truth_relevance: Map of {chunk_id: relevance_grade (e.g. 1 or 2)}.
        k_values: Tuple/list of K thresholds to evaluate.

    Returns:
        Dict of metric names to calculated float scores.
    """
    relevant_ids = {cid for cid, rel in ground_truth_relevance.items() if rel > 0}
    metrics: dict[str, float] = {
        "mrr": round(calculate_mrr(retrieved_ids, relevant_ids), 4),
    }

    for k in k_values:
        metrics[f"hit_rate@{k}"] = round(calculate_hit_rate(retrieved_ids, relevant_ids, k), 4)
        metrics[f"recall@{k}"] = round(calculate_recall_at_k(retrieved_ids, relevant_ids, k), 4)
        metrics[f"precision@{k}"] = round(calculate_precision_at_k(retrieved_ids, relevant_ids, k), 4)
        metrics[f"ndcg@{k}"] = round(calculate_ndcg_at_k(retrieved_ids, ground_truth_relevance, k), 4)

    return metrics
