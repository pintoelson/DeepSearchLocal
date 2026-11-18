"""Unit tests for rag.embeddings (CPUEmbeddingPipeline)."""

import pytest

from rag.embeddings import CPUEmbeddingPipeline


@pytest.fixture(scope="module")
def embedding_pipeline():
    """Shared CPU embedding pipeline fixture."""
    return CPUEmbeddingPipeline()


def test_dense_embeddings_shape(embedding_pipeline):
    """Verify dense embeddings produce normalized 384-dimensional vectors."""
    texts = [
        "RTX 3050 has 6GB of VRAM.",
        "DeepSearch-Local uses a 3B SLM with LangGraph.",
    ]
    dense_vecs = embedding_pipeline.embed_chunks_dense(texts)

    assert len(dense_vecs) == 2
    assert len(dense_vecs[0]) == 384
    assert len(dense_vecs[1]) == 384
    assert all(isinstance(val, float) for val in dense_vecs[0])


def test_sparse_bm25_embeddings(embedding_pipeline):
    """Verify sparse embeddings produce valid index-value pairs."""
    texts = [
        "Okapi BM25 uses term frequency saturation and document length normalization.",
    ]
    sparse_vecs = embedding_pipeline.embed_chunks_sparse(texts)

    assert len(sparse_vecs) == 1
    s_vec = sparse_vecs[0]
    assert "indices" in s_vec
    assert "values" in s_vec
    assert len(s_vec["indices"]) > 0
    assert len(s_vec["indices"]) == len(s_vec["values"])
    assert all(val > 0.0 for val in s_vec["values"])


def test_query_embeddings(embedding_pipeline):
    """Verify query dense and sparse embeddings work properly."""
    query = "What is the VRAM budget for RTX 3050?"
    dense_q = embedding_pipeline.embed_query_dense(query)
    sparse_q = embedding_pipeline.embed_query_sparse(query)

    assert len(dense_q) == 384
    assert len(sparse_q["indices"]) > 0
    assert len(sparse_q["indices"]) == len(sparse_q["values"])


def test_empty_embeddings(embedding_pipeline):
    """Verify empty inputs return expected zero/empty structures without errors."""
    assert embedding_pipeline.embed_chunks_dense([]) == []
    assert embedding_pipeline.embed_chunks_sparse([]) == []

    dense_empty = embedding_pipeline.embed_query_dense("")
    assert len(dense_empty) == 384
    assert all(val == 0.0 for val in dense_empty)

    sparse_empty = embedding_pipeline.embed_query_sparse("")
    assert sparse_empty == {"indices": [], "values": []}
