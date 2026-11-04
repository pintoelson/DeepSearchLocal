"""Comprehensive unit and integration tests for the DeepSearch evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.llm_judge import FallbackJudge, MockJudge, _parse_judge_json
from eval.metrics import (
    calculate_query_diversity,
    evaluate_gap_reflection,
    evaluate_grounded_synthesis,
    evaluate_query_decomposition,
    evaluate_tool_calling,
    extract_json_payload,
    extract_queries_from_text,
)
from eval.ragas_evaluator import RagasEvaluator
from eval.run_benchmark import load_benchmark_dataset, run_benchmarks


class TestMetrics:
    def test_extract_json_payload_direct(self):
        text = '{"tool": "web_search", "arguments": {"query": "bitnet"}}'
        payload = extract_json_payload(text)
        assert isinstance(payload, dict)
        assert payload["tool"] == "web_search"

    def test_extract_json_payload_markdown_fenced(self):
        text = 'Here is the tool call:\n```json\n{\n  "tool": "web_search",\n  "arguments": {"query": "bitnet"}\n}\n```\nDone.'
        payload = extract_json_payload(text)
        assert isinstance(payload, dict)
        assert payload["tool"] == "web_search"

    def test_evaluate_tool_calling_success(self):
        text = '```json\n{"tool": "web_search", "arguments": {"query": "solid state batteries"}}\n```'
        result = evaluate_tool_calling(text, allowed_tools=["web_search", "fetch_page_markdown"])
        assert result["is_valid_json"] is True
        assert result["has_valid_schema"] is True
        assert result["tool_name"] == "web_search"
        assert result["is_allowed_tool"] is True
        assert result["score"] == 1.0

    def test_evaluate_tool_calling_invalid(self):
        text = "I cannot call a tool right now."
        result = evaluate_tool_calling(text)
        assert result["is_valid_json"] is False
        assert result["score"] == 0.0

    def test_evaluate_grounded_synthesis_exact_citations(self):
        text = (
            "BitNet b1.58 uses ternary weights [^chunk_101] and reduces energy by 71.4% [^chunk_102]. "
            "Memory footprint drops by 2.4x [^chunk_103]."
        )
        context = ["chunk_101", "chunk_102", "chunk_103"]
        result = evaluate_grounded_synthesis(text, context_chunks=context)
        assert result["has_citations"] is True
        assert result["citation_count"] == 3
        assert result["citation_precision"] == 1.0
        assert len(result["hallucinated_citations"]) == 0
        assert result["score"] >= 0.9

    def test_evaluate_grounded_synthesis_hallucinated_citations(self):
        text = "Fact one [^chunk_101], fact two [^chunk_fake999]."
        context = ["chunk_101"]
        result = evaluate_grounded_synthesis(text, context_chunks=context)
        assert result["citation_precision"] == 0.5
        assert "chunk_fake999" in result["hallucinated_citations"]

    def test_evaluate_grounded_synthesis_bracket_fallback(self):
        text = "BitNet uses ternary weights [1] and saves energy."
        result = evaluate_grounded_synthesis(text, context_chunks=["chunk_101"])
        assert result["used_standard_format"] is False
        assert result["used_bracket_fallback"] is True
        assert result["score"] == 0.25

    def test_extract_queries_from_text_numbered(self):
        text = (
            '1. "BitNet b1.58 ternary architecture"\n'
            '2. "Energy efficiency comparison vs FP16"\n'
            '3. "Hardware implementations on FPGAs"'
        )
        queries = extract_queries_from_text(text)
        assert len(queries) == 3
        assert "BitNet b1.58 ternary architecture" in queries

    def test_query_diversity(self):
        identical = ["bitnet b1.58 ternary architecture", "bitnet b1.58 ternary architecture"]
        assert calculate_query_diversity(identical) == 0.0

        orthogonal = [
            "solid state battery sulfide electrolytes",
            "automotive manufacturing pilot line timelines",
            "dendrite prevention polymer separators",
        ]
        assert calculate_query_diversity(orthogonal) > 0.7

    def test_evaluate_query_decomposition(self):
        text = (
            '1. "BitNet b1.58 ternary architecture"\n'
            '2. "Energy efficiency comparison vs FP16"\n'
            '3. "Hardware implementations on FPGAs"'
        )
        result = evaluate_query_decomposition(text, target_min=3, target_max=5)
        assert result["query_count"] == 3
        assert result["is_in_target_range"] is True
        assert result["score"] > 0.7

    def test_evaluate_gap_reflection(self):
        text = (
            "Gaps Identified: The section lacks details on scaling laws and hardware accelerators.\n"
            "Follow-up Queries:\n"
            '1. "BitNet b1.58 scaling laws empirical data"\n'
            '2. "BitNet FPGA accelerator latency"'
        )
        result = evaluate_gap_reflection(text, expected_gaps=["scaling laws", "hardware accelerators"])
        assert result["has_gap_analysis"] is True
        assert result["has_followup_queries"] is True
        assert result["expected_aspect_overlap"] == 1.0
        assert result["score"] == 1.0


class TestLLMJudge:
    def test_mock_judge(self):
        judge = MockJudge()
        eval_res = judge.evaluate_pair(
            task="grounded_synthesis",
            topic="BitNet",
            user_prompt="Summarize BitNet",
            response_base="BitNet is 1-bit.",
            response_ft="BitNet is 1-bit [^chunk_01].",
        )
        assert eval_res["winner"] == "finetuned"
        assert eval_res["ft_score"] > eval_res["base_score"]

    def test_fallback_judge_force_mock(self):
        judge = FallbackJudge(force_mock=True)
        eval_res = judge.evaluate_pair(
            task="tool_calling",
            topic="Search BitNet",
            user_prompt="Search for BitNet",
            response_base="Search BitNet on Google",
            response_ft='{"tool": "web_search", "arguments": {"query": "BitNet"}}',
        )
        assert eval_res["winner"] == "finetuned"

    def test_parse_judge_json(self):
        raw = '```json\n{"base_score": 2.5, "ft_score": 4.5, "winner": "finetuned", "reasoning": "Model B used valid JSON."}\n```'
        parsed = _parse_judge_json(raw)
        assert parsed["base_score"] == 2.5
        assert parsed["ft_score"] == 4.5
        assert parsed["winner"] == "finetuned"


class TestDatasetAndRagas:
    def test_dataset_loading_and_filtering(self):
        items = load_benchmark_dataset("./eval/test_dataset_tasks.jsonl", limit=2)
        assert len(items) == 8  # 2 items * 4 tasks
        for item in items:
            assert "id" in item
            assert "task" in item
            assert "user_prompt" in item

    def test_ragas_evaluator(self):
        evaluator = RagasEvaluator(force_mock=True)
        report = "BitNet reduces energy consumption by 71.4% [^chunk_01]."
        ground_truth = [{"chunk_id": "chunk_01"}, {"chunk_id": "chunk_02"}]
        metrics = evaluator.evaluate(
            report=report,
            ground_truth_sources=ground_truth,
            expected_outline_topics=["BitNet", "Energy"],
        )
        assert metrics["citation_precision"] == 1.0
        assert metrics["chunk_coverage"] == 0.5
        assert metrics["outline_coverage"] == 1.0


class TestBenchmarkRunner:
    def test_mock_benchmark_run(self, tmp_path):
        res = run_benchmarks(
            backend="mock",
            dataset_path="./eval/test_dataset_tasks.jsonl",
            tasks=["tool_calling", "grounded_synthesis"],
            limit=1,
            use_judge=False,
            output_dir=str(tmp_path),
        )
        assert res["summary"]["total_samples"] == 2
        assert "tool_calling" in res["summary"]["tasks"]
        assert "grounded_synthesis" in res["summary"]["tasks"]
        assert Path(res["json_path"]).exists()
        assert Path(res["md_path"]).exists()
