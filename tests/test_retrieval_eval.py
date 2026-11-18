"""Unit tests for retrieval evaluation metrics and benchmark runner."""

import math
import pytest

from eval.retrieval_metrics import (
    calculate_hit_rate,
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_dcg_at_k,
    calculate_ndcg_at_k,
    evaluate_retrieval_result,
)
from eval.run_retrieval_benchmark import run_benchmarks


class TestRetrievalMetrics:
    def test_calculate_hit_rate(self):
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = {"doc_3"}

        assert calculate_hit_rate(retrieved, relevant, k=2) == 0.0
        assert calculate_hit_rate(retrieved, relevant, k=3) == 1.0
        assert calculate_hit_rate(retrieved, set(), k=5) == 0.0
        assert calculate_hit_rate(retrieved, relevant, k=0) == 0.0

    def test_calculate_recall_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = {"doc_2", "doc_4"}

        # At k=1: 0/2
        assert calculate_recall_at_k(retrieved, relevant, k=1) == 0.0
        # At k=2: 1/2
        assert calculate_recall_at_k(retrieved, relevant, k=2) == 0.5
        # At k=4: 2/2
        assert calculate_recall_at_k(retrieved, relevant, k=4) == 1.0

    def test_calculate_precision_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = {"doc_2", "doc_4"}

        # At k=2: 1 hit in 2 items = 0.5
        assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.5
        # At k=4: 2 hits in 4 items = 0.5
        assert calculate_precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_calculate_mrr(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        assert calculate_mrr(retrieved, {"doc_1"}) == 1.0
        assert calculate_mrr(retrieved, {"doc_2"}) == 0.5
        assert calculate_mrr(retrieved, {"doc_3"}) == pytest.approx(1.0 / 3.0)
        assert calculate_mrr(retrieved, {"doc_nonexistent"}) == 0.0

    def test_calculate_ndcg_at_k(self):
        # Perfect ranking: highest relevance first
        retrieved = ["doc_a", "doc_b", "doc_c"]
        rel_map = {"doc_a": 2, "doc_b": 1, "doc_c": 0}
        assert calculate_ndcg_at_k(retrieved, rel_map, k=3) == 1.0

        # Sub-optimal ranking: lower relevance first
        sub_retrieved = ["doc_b", "doc_a", "doc_c"]
        ndcg_sub = calculate_ndcg_at_k(sub_retrieved, rel_map, k=3)
        assert 0.0 < ndcg_sub < 1.0

    def test_evaluate_retrieval_result_structure(self):
        retrieved = ["doc_1", "doc_2"]
        ground_truth = {"doc_2": 2}
        result = evaluate_retrieval_result(retrieved, ground_truth, k_values=(1, 2))

        assert "mrr" in result
        assert "hit_rate@1" in result
        assert "recall@2" in result
        assert result["hit_rate@1"] == 0.0
        assert result["hit_rate@2"] == 1.0
        assert result["mrr"] == 0.5


def test_run_retrieval_benchmark_integration(tmp_path):
    """Verify end-to-end execution of run_benchmarks with temporary output dir."""
    results = run_benchmarks(
        dataset_path="./eval/retrieval_benchmark_dataset.json",
        output_dir=str(tmp_path),
        top_k=5,
        prefetch_pool=10,
    )

    assert len(results) == 4
    mode_names = [r["mode"] for r in results]
    assert "dense_only" in mode_names
    assert "sparse_only" in mode_names
    assert "hybrid_rrf" in mode_names
    assert "hybrid_rrf_reranked" in mode_names

    for r in results:
        assert r["query_count"] == 24
        assert "mrr" in r["overall_metrics"]
        assert "hit_rate@5" in r["overall_metrics"]
        assert r["latency"]["mean_ms"] > 0
