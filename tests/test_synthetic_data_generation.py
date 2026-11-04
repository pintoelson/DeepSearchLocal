"""Unit and integration tests for synthetic trajectory generation and quality gating."""

import json
import tempfile
from pathlib import Path

import pytest

from finetuning.dataset import validate_dataset_schema
from finetuning.generate_synthetic_data import (
    TASK_GAP_REFLECTION,
    TASK_GROUNDED_SYNTHESIS,
    TASK_QUERY_DECOMPOSITION,
    TASK_TOOL_CALLING,
    GeminiClient,
    GenerationCheckpoint,
    OpenRouterClient,
    RWTHGPTClient,
    clean_json_payload,
    build_batch_generation_prompt,
    extract_chunk_ids_from_prompt,
    extract_citations,
    generate_trajectories,
    load_seed_topics,
    merge_datasets,
    parse_batch_response,
    validate_single_trajectory,
)


def test_seed_topics_loading():
    """Verify seed_topics.yaml loads with rich domains and subtopics."""
    config_path = Path("finetuning/configs/seed_topics.yaml")
    assert config_path.is_file(), "seed_topics.yaml must exist"

    seeds = load_seed_topics(config_path)
    assert len(seeds) >= 20, f"Expected at least 20 topic seeds, found {len(seeds)}"

    domain_ids = {s.domain_id for s in seeds}
    assert "ai_ml_systems" in domain_ids
    assert "materials_energy" in domain_ids
    assert "physics_quantum" in domain_ids
    assert "biotech_genomics" in domain_ids
    assert "computer_systems" in domain_ids

    for s in seeds:
        assert s.topic, "Topic must not be empty"
        assert len(s.subtopics) >= 2, f"Expected >= 2 subtopics for topic: {s.topic}"


def test_validate_task1_query_decomposition():
    """Verify quality gate logic for Task 1: Query Decomposition."""
    # Valid
    valid_rec = {
        "messages": [
            {"role": "system", "content": "You are DeepSearch-Local."},
            {"role": "user", "content": "Decompose: 1-bit LLMs."},
            {
                "role": "assistant",
                "content": json.dumps({"queries": ["BitNet b1.58 scaling", "Ternary weights hardware"]}),
            },
        ]
    }
    res = validate_single_trajectory(valid_rec, TASK_QUERY_DECOMPOSITION)
    assert res.is_valid is True

    # Invalid: Not JSON
    bad_json = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "Here are some queries: 1. Bitnet 2. LLMs"},
        ]
    }
    assert validate_single_trajectory(bad_json, TASK_QUERY_DECOMPOSITION).is_valid is False

    # Invalid: Only 1 query
    single_q = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps({"queries": ["Only one query"]})},
        ]
    }
    assert validate_single_trajectory(single_q, TASK_QUERY_DECOMPOSITION).is_valid is False


def test_validate_task2_tool_calling():
    """Verify quality gate logic for Task 2: Tool Calling."""
    # Valid web_search
    valid_search = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Search for solid-state batteries."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool": "web_search", "arguments": {"query": "solid state battery electrolytes", "max_results": 5}}
                ),
            },
        ]
    }
    assert validate_single_trajectory(valid_search, TASK_TOOL_CALLING).is_valid is True

    # Valid fetch_page_markdown
    valid_scrape = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Fetch page content."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool": "fetch_page_markdown", "arguments": {"url": "https://example.com/paper"}}
                ),
            },
        ]
    }
    assert validate_single_trajectory(valid_scrape, TASK_TOOL_CALLING).is_valid is True

    # Invalid tool
    bad_tool = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Execute"},
            {"role": "assistant", "content": json.dumps({"tool": "unauthorized_bash", "arguments": {"cmd": "ls"}})},
        ]
    }
    assert validate_single_trajectory(bad_tool, TASK_TOOL_CALLING).is_valid is False

    # Missing query in web_search
    missing_query = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Search"},
            {"role": "assistant", "content": json.dumps({"tool": "web_search", "arguments": {"max_results": 5}})},
        ]
    }
    assert validate_single_trajectory(missing_query, TASK_TOOL_CALLING).is_valid is False


def test_validate_task3_grounded_synthesis():
    """Verify quality gate logic for Task 3: Grounded Synthesis with Citation Verification."""
    user_context = (
        "Context:\n"
        "[Chunk c_101]: Ternary weights reduce energy consumption by up to 8.9x.\n"
        "[Chunk c_202]: Integer addition replaces matrix multiplications in BitNet.\n\n"
        "Summarize the findings."
    )

    # Valid synthesis with accurate citations
    valid_synth = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user_context},
            {
                "role": "assistant",
                "content": (
                    "Ternary quantization yields an 8.9x reduction in operational energy [^c_101]. "
                    "This efficiency stems from replacing floating-point operations with integer addition [^c_202]."
                ),
            },
        ]
    }
    assert validate_single_trajectory(valid_synth, TASK_GROUNDED_SYNTHESIS).is_valid is True

    # Invalid: Hallucinated citation not in context
    hallucinated_synth = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user_context},
            {
                "role": "assistant",
                "content": "BitNet reduces power consumption drastically [^c_101] and scales well [^fake_chunk_999].",
            },
        ]
    }
    res_fake = validate_single_trajectory(hallucinated_synth, TASK_GROUNDED_SYNTHESIS)
    assert res_fake.is_valid is False
    assert "hallucinated chunk 'fake_chunk_999'" in res_fake.error_message

    # Invalid: Missing any citations
    no_citations = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user_context},
            {
                "role": "assistant",
                "content": "Ternary quantization yields an 8.9x reduction in operational energy without any source citation.",
            },
        ]
    }
    assert validate_single_trajectory(no_citations, TASK_GROUNDED_SYNTHESIS).is_valid is False


def test_validate_task4_gap_reflection():
    """Verify quality gate logic for Task 4: Gap Detection & Reflection."""
    valid_gap = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Topic: Solid-state batteries\nDraft: Section 1 drafted."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "has_gaps": True,
                        "sections_to_revise": ["manufacturing_timeline"],
                        "follow_up_queries": ["pilot line manufacturing solid-state batteries"],
                    }
                ),
            },
        ]
    }
    assert validate_single_trajectory(valid_gap, TASK_GAP_REFLECTION).is_valid is True

    # Invalid: Missing has_gaps
    missing_key = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps({"sections_to_revise": ["s1"]})},
        ]
    }
    assert validate_single_trajectory(missing_key, TASK_GAP_REFLECTION).is_valid is False


def test_citation_and_chunk_extraction():
    """Verify chunk ID and citation regex extraction helpers."""
    text = "Here is evidence [^chunk_a1] and another citation [^chunk-b2_99]."
    cites = extract_citations(text)
    assert cites == ["chunk_a1", "chunk-b2_99"]

    prompt = (
        "Retrieved context:\n"
        "[Chunk a_01]: First passage\n"
        "[Chunk b_02]: Second passage\n"
        "[c_03]: Third passage format"
    )
    chunks = extract_chunk_ids_from_prompt(prompt)
    assert "a_01" in chunks
    assert "b_02" in chunks
    assert "c_03" in chunks


def test_parse_batch_response():
    """Verify batch parser handles JSON arrays, markdown fenced blocks, and fallback."""
    raw_array = json.dumps([{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi"}]}])
    assert len(parse_batch_response(raw_array)) == 1

    markdown_fenced = f"Here is the result:\n```json\n{raw_array}\n```\nHope this helps!"
    assert len(parse_batch_response(markdown_fenced)) == 1


def test_dry_run_generation_and_checkpoint_resume():
    """Verify end-to-end dry-run generation, schema validation, and checkpoint resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "synthetic_dry_run.jsonl"
        ckpt_path = Path(tmpdir) / "checkpoint.json"

        # 1. Generate 10 samples
        res_file = generate_trajectories(
            output_path=str(out_path),
            count=10,
            batch_size=5,
            dry_run=True,
            checkpoint_path=str(ckpt_path),
        )
        assert res_file.is_file()

        # Validate format
        is_valid, err, count = validate_dataset_schema(res_file)
        assert is_valid is True, f"Generated dataset invalid: {err}"
        assert count == 10

        # Verify checkpoint
        ckpt = GenerationCheckpoint.load(ckpt_path)
        assert ckpt.total_valid_samples == 10
        assert sum(ckpt.task_counts.values()) == 10

        # 2. Resume to 16 samples
        generate_trajectories(
            output_path=str(out_path),
            count=16,
            batch_size=4,
            dry_run=True,
            checkpoint_path=str(ckpt_path),
        )

        is_valid2, _, count2 = validate_dataset_schema(res_file)
        assert is_valid2 is True
        assert count2 == 16


def test_gemini_client_interface():
    """Verify GeminiClient initialization, quota report, and missing key error handling."""
    client = GeminiClient(api_key="mock_key")
    quota = client.check_quota()
    assert quota["provider"] == "google_ai_studio"
    assert quota["free_requests"]["limit"] == 1500

    # Missing key error
    empty_client = GeminiClient(api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY is not set"):
        empty_client.generate_chat_completion([{"role": "user", "content": "hello"}])


def test_merge_datasets():
    """Verify merge_datasets combines files, filters malformed lines, and deduplicates identical samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        f1 = tmp / "gemini.jsonl"
        f2 = tmp / "openrouter.jsonl"
        out = tmp / "merged.jsonl"

        # f1 has 2 valid samples
        rec1 = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Decompose: 1-bit LLMs"},
                {"role": "assistant", "content": json.dumps({"queries": ["query 1", "query 2"]})},
            ]
        }
        rec2 = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Decompose: Solid state batteries"},
                {"role": "assistant", "content": json.dumps({"queries": ["battery alpha", "battery beta"]})},
            ]
        }
        f1.write_text(json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n")

        # f2 has rec2 (duplicate), rec3 (new), and a malformed record
        rec3 = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Decompose: Quantum computing"},
                {"role": "assistant", "content": json.dumps({"queries": ["quantum state", "quantum gates"]})},
            ]
        }
        f2.write_text(json.dumps(rec2) + "\n" + json.dumps(rec3) + "\n{ bad json }\n")

        merged_path = merge_datasets([f1, f2], out)
        assert merged_path.is_file()

        is_valid, err, count = validate_dataset_schema(merged_path)
        assert is_valid is True
        # rec1 + rec2 + rec3 = 3 unique valid records
        assert count == 3


def test_clean_json_payload_and_markdown_wrapping():
    """Verify clean_json_payload removes code fences and validate_single_trajectory accepts wrapped JSON."""
    raw_wrapped = "```json\n{\n  \"queries\": [\"quantum state\", \"quantum gates\"]\n}\n```"
    cleaned = clean_json_payload(raw_wrapped)
    assert cleaned.startswith("{") and cleaned.endswith("}")

    rec = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Decompose: Quantum computing"},
            {"role": "assistant", "content": raw_wrapped},
        ]
    }
    res = validate_single_trajectory(rec, TASK_QUERY_DECOMPOSITION)
    assert res.is_valid is True


def test_rwthgpt_client_routing_and_quotas():
    """Verify RWTHGPTClient task specialization, quota budgets, and round-robin fallback."""
    client = RWTHGPTClient(
        api_key="mock_key",
        luna_budget=2,
        sol_budget=1,
    )

    # 1. Synthesis tasks route to Luna first
    assert client.select_model_for_task(TASK_GROUNDED_SYNTHESIS) == "gpt-6-luna"
    client.request_counts["gpt-6-luna"] += 1
    assert client.select_model_for_task(TASK_GROUNDED_SYNTHESIS) == "gpt-6-luna"
    client.request_counts["gpt-6-luna"] += 1

    # 2. When Luna budget is hit, route to Sol
    assert client.select_model_for_task(TASK_GROUNDED_SYNTHESIS) == "gpt-6-sol"
    client.request_counts["gpt-6-sol"] += 1

    # 3. When Sol budget is hit, fallback to unlimited pool (round-robin)
    first_fallback = client.select_model_for_task(TASK_GROUNDED_SYNTHESIS)
    assert first_fallback in client.UNLIMITED_MODELS

    # 4. Decomposition tasks route to unlimited round-robin
    m1 = client.select_model_for_task(TASK_QUERY_DECOMPOSITION)
    m2 = client.select_model_for_task(TASK_QUERY_DECOMPOSITION)
    assert m1 in client.UNLIMITED_MODELS
    assert m2 in client.UNLIMITED_MODELS
    assert m1 != m2  # Round-robin advances

    # 5. Check quota reports limits correctly
    q = client.check_quota()
    assert q["limits"]["gpt-6-luna"]["hard_limit"] == 250
    assert q["limits"]["gpt-6-sol"]["hard_limit"] == 60
