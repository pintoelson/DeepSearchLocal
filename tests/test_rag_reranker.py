"""Unit tests for rag.reranker (CrossEncoderReranker)."""

import pytest

from rag.chunking import RecursiveMarkdownChunker
from rag.models import DocumentChunk, ScoredChunk
from rag.reranker import CrossEncoderReranker


@pytest.fixture(scope="module")
def reranker():
    """Module-level CrossEncoderReranker fixture."""
    return CrossEncoderReranker()


def test_cross_encoder_reranks_and_filters_top_n(reranker):
    """Verify cross-encoder scores candidate chunks and filters down to top_n."""
    chunker = RecursiveMarkdownChunker()

    docs = [
        (
            "# Bread\nSourdough bread is made by fermenting dough with wild yeast.",
            "https://example.com/1",
        ),
        ("# Cooking\nPasta is made from unleavened dough of wheat flour.", "https://example.com/2"),
        (
            "# Memory\nThe RTX 3050 has 6GB VRAM which limits the KV cache size.",
            "https://example.com/3",
        ),
        (
            "# GPU Cores\nAmpere architecture has streaming multiprocessors.",
            "https://example.com/4",
        ),
    ]

    candidates = []
    for i, (text, url) in enumerate(docs):
        chunks = chunker.chunk_document(text, url)
        candidates.append(ScoredChunk(chunk=chunks[0], rrf_score=1.0 / (60 + i)))

    query = "How much GPU VRAM does the RTX 3050 have for model context?"
    reranked = reranker.rerank(query=query, candidates=candidates, top_n=2)

    assert len(reranked) == 2

    # The memory chunk must be ranked #1
    top_chunk = reranked[0]
    assert "RTX 3050 has 6GB VRAM" in top_chunk.chunk.raw_content
    assert top_chunk.reranker_score is not None
    assert 0.0 <= top_chunk.reranker_score <= 1.0

    # Scores must be strictly descending
    assert reranked[0].reranker_score >= reranked[1].reranker_score

    # Sourdough and cooking must NOT be in top 2
    assert "Sourdough" not in reranked[0].chunk.raw_content
    assert "Sourdough" not in reranked[1].chunk.raw_content


def test_cross_encoder_empty_inputs(reranker):
    """Verify empty candidates or top_n <= 0 returns empty list gracefully."""
    dummy_chunk = DocumentChunk(
        chunk_id="test_id",
        source_url="https://example.com",
        raw_content="test content",
        embedding_text="test content",
    )
    assert reranker.rerank("query", [], top_n=4) == []
    assert reranker.rerank("query", [ScoredChunk(chunk=dummy_chunk, rrf_score=1.0)], top_n=0) == []
