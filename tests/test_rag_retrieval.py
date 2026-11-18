"""Unit tests for rag.hybrid_retriever (HybridRetriever with server-side RRF)."""

import pytest

from rag.chunking import RecursiveMarkdownChunker
from rag.hybrid_retriever import HybridRetriever
from rag.vector_store import EphemeralVectorStore


@pytest.fixture
def hybrid_setup():
    """Set up an in-memory vector store with indexed documents and a retriever."""
    store = EphemeralVectorStore(use_memory=True)
    retriever = HybridRetriever(vector_store=store, rrf_k=60)
    session_id = "test_retrieval_sess"

    doc1 = """# RTX 3050 Laptop GPU Specifications
The NVIDIA GeForce RTX 3050 Laptop GPU features 6144 MiB of GDDR6 VRAM.
It operates on an Ampere GA107 architecture with 2048 CUDA cores and 64 Tensor cores.
"""
    doc2 = """# Small Language Models (SLMs)
A 3B parameter model like Qwen2.5-3B-Instruct quantized to 4-bit requires ~2.0 GB VRAM.
The 4096-token KV cache occupies approximately 1.0 GB of VRAM.
"""
    doc3 = """# Sourdough Bread Baking
Sourdough bread is made by the fermentation of dough using wild lactobacillaceae and yeast.
Lactic acid produced by lactobacilli gives it a sour taste.
"""
    chunker = RecursiveMarkdownChunker()
    chunks1 = chunker.chunk_document(doc1, "https://example.com/gpu", "GPU Specs")
    chunks2 = chunker.chunk_document(doc2, "https://example.com/slm", "SLM Specs")
    chunks3 = chunker.chunk_document(doc3, "https://example.com/bread", "Bread Baking")

    store.add_documents(session_id, chunks1 + chunks2 + chunks3)

    yield session_id, retriever, store

    # Teardown
    store.delete_session_collection(session_id)


def test_hybrid_retrieval_finds_relevant_chunks(hybrid_setup):
    """Verify hybrid retrieval ranks relevant technical chunks at the top."""
    session_id, retriever, _ = hybrid_setup

    query = "How much VRAM does the RTX 3050 have for running SLMs?"
    results = retriever.retrieve(session_id=session_id, query=query, top_k=5)

    assert len(results) > 0
    # Top result should mention RTX 3050 or VRAM
    top_content = results[0].chunk.raw_content
    assert "RTX 3050" in top_content or "VRAM" in top_content
    assert results[0].rrf_score > 0.0

    # Verify sourdough bread is ranked lowest or not in top matches
    assert "Sourdough" not in results[0].chunk.raw_content


def test_hybrid_retrieval_exact_keyword_match(hybrid_setup):
    """Verify BM25 sparse component boosts exact identifier matches."""
    session_id, retriever, _ = hybrid_setup

    query = "GA107 Ampere CUDA cores"
    results = retriever.retrieve(session_id=session_id, query=query, top_k=2)

    assert len(results) > 0
    assert "GA107" in results[0].chunk.raw_content


def test_hybrid_retrieval_empty_query(hybrid_setup):
    """Verify empty query returns empty list gracefully."""
    session_id, retriever, _ = hybrid_setup
    assert retriever.retrieve(session_id, "") == []
    assert retriever.retrieve(session_id, "   ") == []


def test_hybrid_retrieval_nonexistent_session(hybrid_setup):
    """Verify querying non-existent session returns empty list gracefully."""
    _, retriever, _ = hybrid_setup
    results = retriever.retrieve(session_id="nonexistent_id", query="test")
    assert results == []
