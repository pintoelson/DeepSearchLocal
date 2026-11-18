"""End-to-end integration tests for LangGraph cyclic deep research state machine."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.graph import build_research_graph
from agent.llm import MockAgentLLM
from agent.state import ResearchState
from mcp_server.tools.models import SearchResult
from rag.models import DocumentChunk, ScoredChunk


@pytest.mark.asyncio
async def test_graph_e2e_single_iteration_pass():
    """Verify single-pass graph execution when reflection finds no gaps."""

    planner_resp = """{
      "outline": ["1. Executive Summary", "2. Technical Analysis"],
      "search_queries": ["test topic query"]
    }"""

    synth_resp = "This is a detailed analysis citing chunk [^test_chunk_1]."

    reflect_resp = """{
      "has_gaps": false,
      "sections_to_revise": [],
      "unresolved_gaps": [],
      "follow_up_queries": []
    }"""

    def mock_llm_handler(messages):
        system_content = messages[0]["content"] if messages else ""
        if "expert research planner" in system_content:
            return planner_resp
        elif "scientific research synthesizer" in system_content:
            return synth_resp
        elif "critical scientific research auditor" in system_content:
            return reflect_resp
        return "{}"

    mock_llm = MockAgentLLM(custom_handler=mock_llm_handler)

    doc_chunk = DocumentChunk(
        chunk_id="test_chunk_1",
        source_url="https://example.com/test",
        page_title="Example Test Page",
        heading_path="Overview",
        raw_content="Empirical verification content for testing.",
        embedding_text="Empirical verification content for testing.",
        token_count=15,
        chunk_index=0,
    )

    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        ScoredChunk(chunk=doc_chunk, hybrid_score=0.95, dense_score=0.9, sparse_score=0.9)
    ]
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        ScoredChunk(chunk=doc_chunk, hybrid_score=0.95, dense_score=0.9, sparse_score=0.9, reranker_score=0.98)
    ]

    mock_search_results = [
        SearchResult(title="Example", url="https://example.com/test", snippet="snippet")
    ]

    graph = build_research_graph(
        llm=mock_llm,
        vector_store=mock_store,
        retriever=mock_retriever,
        reranker=mock_reranker,
    )

    with patch("agent.nodes.search.web_search", new_callable=AsyncMock) as mock_search, \
         patch("agent.nodes.search.fetch_page_markdown", new_callable=AsyncMock) as mock_scrape:
        mock_search.return_value = mock_search_results
        mock_scrape.return_value = "# Example Title\nArticle body."

        initial_state = ResearchState(
            research_topic="Autonomous Agent Design",
            retain_collection=True,
        )

        final_state = await graph.ainvoke(initial_state)

        assert final_state["final_report"] is not None
        assert "# Autonomous Agent Design" in final_state["final_report"]
        assert "## 1. Executive Summary" in final_state["final_report"]
        assert "[1]" in final_state["final_report"]
        assert "## References" in final_state["final_report"]
        assert "[1] **Example Test Page** - [https://example.com/test](https://example.com/test)" in final_state["final_report"]
        assert final_state["iteration_count"] == 1


@pytest.mark.asyncio
async def test_graph_e2e_cyclic_self_correction():
    """Verify cyclic flow where reflection triggers a revision loop back to search."""

    planner_resp = """{
      "outline": ["1. Core Findings"],
      "search_queries": ["initial query"]
    }"""

    synth_initial = "Preliminary findings without enough data [^chunk_a]."
    synth_revised = "Enriched findings with complete data [^chunk_a] and [^chunk_b]."

    reflect_first_pass = """{
      "has_gaps": true,
      "sections_to_revise": ["1. Core Findings"],
      "unresolved_gaps": ["Need benchmark metrics"],
      "follow_up_queries": ["benchmark metrics query"]
    }"""

    reflect_second_pass = """{
      "has_gaps": false,
      "sections_to_revise": [],
      "unresolved_gaps": [],
      "follow_up_queries": []
    }"""

    call_counts = {"reflect": 0, "synth": 0}

    def mock_llm_handler(messages):
        system_content = messages[0]["content"] if messages else ""
        if "expert research planner" in system_content:
            return planner_resp
        elif "scientific research synthesizer" in system_content:
            call_counts["synth"] += 1
            if call_counts["synth"] == 1:
                return synth_initial
            return synth_revised
        elif "critical scientific research auditor" in system_content:
            call_counts["reflect"] += 1
            if call_counts["reflect"] == 1:
                return reflect_first_pass
            return reflect_second_pass
        return "{}"

    mock_llm = MockAgentLLM(custom_handler=mock_llm_handler)

    chunk_a = DocumentChunk(
        chunk_id="chunk_a",
        source_url="https://source-a.com",
        page_title="Source A",
        heading_path="H1",
        raw_content="Content A",
        embedding_text="Content A",
        token_count=10,
        chunk_index=0,
    )
    chunk_b = DocumentChunk(
        chunk_id="chunk_b",
        source_url="https://source-b.com",
        page_title="Source B",
        heading_path="H2",
        raw_content="Content B",
        embedding_text="Content B",
        token_count=10,
        chunk_index=0,
    )

    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        ScoredChunk(chunk=chunk_a, hybrid_score=0.9, dense_score=0.9, sparse_score=0.9),
        ScoredChunk(chunk=chunk_b, hybrid_score=0.85, dense_score=0.85, sparse_score=0.85),
    ]
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        ScoredChunk(chunk=chunk_a, hybrid_score=0.9, dense_score=0.9, sparse_score=0.9, reranker_score=0.95),
        ScoredChunk(chunk=chunk_b, hybrid_score=0.85, dense_score=0.85, sparse_score=0.85, reranker_score=0.90),
    ]

    graph = build_research_graph(
        llm=mock_llm,
        vector_store=mock_store,
        retriever=mock_retriever,
        reranker=mock_reranker,
    )

    with patch("agent.nodes.search.web_search", new_callable=AsyncMock) as mock_search, \
         patch("agent.nodes.search.fetch_page_markdown", new_callable=AsyncMock) as mock_scrape:
        mock_search.return_value = [
            SearchResult(title="Source", url="https://source.com", snippet="info")
        ]
        mock_scrape.return_value = "# Source\nText."

        initial_state = ResearchState(
            research_topic="Cyclic Agent Benchmark",
            max_iterations=3,
            retain_collection=True,
        )

        final_state = await graph.ainvoke(initial_state)

        # Reflection ran twice
        assert call_counts["reflect"] == 2
        # Synthesis ran at least twice (initial + revision)
        assert call_counts["synth"] >= 2
        # Final report contains both normalized citations
        assert "[1]" in final_state["final_report"]
        assert "[2]" in final_state["final_report"]
        assert "Source A" in final_state["final_report"]
        assert "Source B" in final_state["final_report"]
        assert final_state["iteration_count"] == 2
