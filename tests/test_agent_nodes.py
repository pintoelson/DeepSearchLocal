"""Unit tests for individual LangGraph agent nodes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.llm import MockAgentLLM
from agent.nodes.ingest import ingestion_node
from agent.nodes.planner import planner_node
from agent.nodes.reflect import reflection_node
from agent.nodes.search import search_scrape_node
from agent.nodes.synth import synthesis_loop
from agent.state import ResearchState
from mcp_server.tools.models import SearchResult
from rag.models import DocumentChunk, ScoredChunk


@pytest.mark.asyncio
async def test_planner_node_success():
    canned_json = """{
      "outline": ["1. Intro", "2. Details"],
      "search_queries": ["query 1", "query 2"]
    }"""
    mock_llm = MockAgentLLM(default_response=canned_json)
    state = ResearchState(research_topic="Quantum Computing")

    result = await planner_node(state, llm=mock_llm)
    assert result["outline"] == ["1. Intro", "2. Details"]
    assert result["search_queries"] == ["query 1", "query 2"]


@pytest.mark.asyncio
async def test_planner_node_fallback_on_error():
    mock_llm = MockAgentLLM(default_response="Corrupted text")
    state = ResearchState(research_topic="Quantum Computing")

    result = await planner_node(state, llm=mock_llm)
    assert len(result["outline"]) >= 4
    assert len(result["search_queries"]) >= 3
    assert any("Quantum Computing" in s for s in result["outline"])


@pytest.mark.asyncio
async def test_search_scrape_node():
    state = ResearchState(
        research_topic="Test Topic",
        search_queries=["test query"],
        scraped_urls={"https://already-scraped.com"},
    )

    mock_search_results = [
        SearchResult(title="Old", url="https://already-scraped.com", snippet="old"),
        SearchResult(title="New Page", url="https://new-source.com", snippet="fresh info"),
    ]

    with patch("agent.nodes.search.web_search", new_callable=AsyncMock) as mock_web_search, \
         patch("agent.nodes.search.fetch_page_markdown", new_callable=AsyncMock) as mock_fetch:
        mock_web_search.return_value = mock_search_results
        mock_fetch.return_value = "# Markdown Article\nUseful content."

        result = await search_scrape_node(state)

        assert "https://new-source.com" in result["scraped_urls"]
        assert len(result["raw_documents"]) == 1
        assert result["raw_documents"][0]["url"] == "https://new-source.com"
        assert result["raw_documents"][0]["title"] == "New Page"


def test_ingestion_node():
    state = ResearchState(
        research_topic="Test Topic",
        session_id="test1234",
        raw_documents=[
            {
                "url": "https://test.com",
                "title": "Test Title",
                "content": "# Heading\nParagraph content for chunking.",
            }
        ],
    )

    mock_vector_store = MagicMock()
    mock_vector_store.add_documents.return_value = 1

    result = ingestion_node(state, vector_store=mock_vector_store)

    mock_vector_store.create_session_collection.assert_called_once_with("test1234")
    mock_vector_store.add_documents.assert_called_once()
    assert len(result["evidence_chunks"]) >= 1
    assert result["raw_documents"] == []


@pytest.mark.asyncio
async def test_synthesis_loop():
    mock_llm = MockAgentLLM(default_response="Drafted section content with [^c8a1f29b].")
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()

    doc_chunk = DocumentChunk(
        chunk_id="c8a1f29b",
        source_url="https://source.com",
        page_title="Title",
        heading_path="H1",
        raw_content="Fact content.",
        embedding_text="Fact content.",
        token_count=10,
        chunk_index=0,
    )
    mock_retriever.retrieve.return_value = [
        ScoredChunk(chunk=doc_chunk, hybrid_score=0.9, dense_score=0.9, sparse_score=0.9)
    ]
    mock_reranker.rerank.return_value = [
        ScoredChunk(chunk=doc_chunk, hybrid_score=0.9, dense_score=0.9, sparse_score=0.9, reranker_score=0.95)
    ]

    state = ResearchState(
        research_topic="Test Topic",
        outline=["1. Section One"],
        evidence_chunks=[doc_chunk.model_dump()],
    )

    result = await synthesis_loop(
        state,
        llm=mock_llm,
        retriever=mock_retriever,
        reranker=mock_reranker,
    )

    assert "1. Section One" in result["sections"]
    assert "Drafted section content with [^c8a1f29b]." in result["sections"]["1. Section One"]


@pytest.mark.asyncio
async def test_reflection_node_detects_gaps():
    reflection_json = """{
      "has_gaps": true,
      "sections_to_revise": ["1. Section One"],
      "unresolved_gaps": ["Missing data on power consumption"],
      "follow_up_queries": ["power consumption benchmark"]
    }"""
    mock_llm = MockAgentLLM(default_response=reflection_json)
    state = ResearchState(
        research_topic="Test Topic",
        outline=["1. Section One"],
        sections={"1. Section One": "Incomplete text."},
        iteration_count=0,
        max_iterations=3,
    )

    result = await reflection_node(state, llm=mock_llm)

    assert result["iteration_count"] == 1
    assert result["sections_to_revise"] == ["1. Section One"]
    assert result["unresolved_gaps"] == ["Missing data on power consumption"]
    assert result["search_queries"] == ["power consumption benchmark"]


@pytest.mark.asyncio
async def test_reflection_node_max_iterations_guardrail():
    mock_llm = MockAgentLLM(default_response="should not be called")
    state = ResearchState(
        research_topic="Test Topic",
        outline=["1. Section One"],
        iteration_count=2,
        max_iterations=2,
    )

    result = await reflection_node(state, llm=mock_llm)

    assert result["iteration_count"] == 2
    assert result["sections_to_revise"] == []
    assert len(mock_llm.call_history) == 0  # LLM call skipped on max iterations reached
