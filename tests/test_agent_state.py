"""Unit tests for ResearchState schema and channel reducers."""

import operator

from agent.state import ResearchState


def test_research_state_defaults():
    state = ResearchState(research_topic="Test Topic")
    assert state.research_topic == "Test Topic"
    assert state.outline == []
    assert state.search_queries == []
    assert state.scraped_urls == set()
    assert state.evidence_chunks == []
    assert state.raw_documents == []
    assert state.sections == {}
    assert state.sections_to_revise == []
    assert state.unresolved_gaps == []
    assert state.iteration_count == 0
    assert state.max_iterations == 2
    assert state.final_report is None
    assert state.citation_map == {}
    assert state.retain_collection is False
    assert len(state.session_id) == 8


def test_channel_reducers_behavior():
    # search_queries operator.add
    q1 = ["query 1", "query 2"]
    q2 = ["query 3"]
    merged_queries = operator.add(q1, q2)
    assert merged_queries == ["query 1", "query 2", "query 3"]

    # scraped_urls set.union
    u1 = {"https://example.com/a", "https://example.com/b"}
    u2 = {"https://example.com/b", "https://example.com/c"}
    merged_urls = set.union(u1, u2)
    assert merged_urls == {"https://example.com/a", "https://example.com/b", "https://example.com/c"}

    # evidence_chunks operator.add
    c1 = [{"chunk_id": "c1"}]
    c2 = [{"chunk_id": "c2"}]
    merged_chunks = operator.add(c1, c2)
    assert merged_chunks == [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
