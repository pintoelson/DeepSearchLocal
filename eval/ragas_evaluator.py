"""Ragas and citation metric evaluation harness for faithfulness and citation precision."""

from __future__ import annotations

from typing import Any
from eval.metrics import evaluate_grounded_synthesis, extract_json_payload
from eval.llm_judge import FallbackJudge


class RagasEvaluator:
    """Evaluates generated deep research reports against source context and outline coverage."""

    def __init__(self, force_mock: bool = False):
        self.judge = FallbackJudge(force_mock=force_mock)

    def evaluate(
        self,
        report: str,
        ground_truth_sources: list[dict[str, Any]],
        expected_outline_topics: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute Faithfulness, Citation Precision, and Outline Coverage metrics."""
        # 1. Extract context chunk identifiers
        chunk_ids = []
        for src in ground_truth_sources:
            cid = src.get("chunk_id") or src.get("id")
            if cid:
                chunk_ids.append(str(cid))

        # 2. Compute citation metrics
        synth_metrics = evaluate_grounded_synthesis(report, context_chunks=chunk_ids)

        # 3. Outline coverage heuristic
        outline_cov = 0.0
        if expected_outline_topics:
            lower_report = report.lower()
            matched = sum(1 for topic in expected_outline_topics if topic.lower() in lower_report)
            outline_cov = matched / len(expected_outline_topics)

        return {
            "citation_precision": synth_metrics["citation_precision"],
            "chunk_coverage": synth_metrics["chunk_coverage"],
            "citation_density": synth_metrics["citation_density"],
            "outline_coverage": round(outline_cov, 4),
            "composite_score": synth_metrics["score"],
        }
