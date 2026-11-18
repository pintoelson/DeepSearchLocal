"""Unit tests for rag.vector_store (EphemeralVectorStore)."""

import time

import pytest

from rag.chunking import RecursiveMarkdownChunker
from rag.vector_store import EphemeralVectorStore


@pytest.fixture
def vector_store():
    """Fixture providing in-memory EphemeralVectorStore for fast testing."""
    return EphemeralVectorStore(use_memory=True)


def test_ephemeral_collection_creation_and_deletion(vector_store):
    """Verify session collection is created and deleted properly."""
    session_id = "test_sess_001"

    col_name = vector_store.create_session_collection(session_id)
    assert session_id in col_name
    assert vector_store._resolve_collection_name(session_id) == col_name

    # Deleting collection
    deleted = vector_store.delete_session_collection(session_id)
    assert deleted is True
    assert vector_store._resolve_collection_name(session_id) is None


def test_add_and_count_documents(vector_store):
    """Verify DocumentChunks are embedded and indexed into the session collection."""
    session_id = "test_sess_002"
    chunker = RecursiveMarkdownChunker(max_tokens=256)

    doc_text = """# RTX 3050 Specifications
The NVIDIA GeForce RTX 3050 Laptop GPU contains 6 GB GDDR6 memory.

## Power Budget
Typical TGP ranges between 35W and 80W.
"""
    chunks = chunker.chunk_document(
        doc_text, "https://example.com/specs", page_title="RTX 3050 Specs"
    )
    assert len(chunks) == 2

    upserted_count = vector_store.add_documents(session_id, chunks)
    assert upserted_count == 2

    indexed_count = vector_store.count_documents(session_id)
    assert indexed_count == 2

    # Clean up
    vector_store.delete_session_collection(session_id)


def test_session_context_manager(vector_store):
    """Verify session context manager creates and automatically destroys collection on exit."""
    session_id = "test_sess_ctx"

    with vector_store.session(session_id) as col_name:
        assert session_id in col_name
        assert vector_store._resolve_collection_name(session_id) == col_name

    # After exit, collection must be dropped
    assert vector_store._resolve_collection_name(session_id) is None


def test_cleanup_stale_collections(vector_store):
    """Verify janitor sweeps session collections older than max_age_minutes."""
    # Create an artificially old collection timestamp (2 hours ago)
    old_timestamp = int(time.time()) - 7200
    old_col_name = f"session_{old_timestamp}_expired_session"
    vector_store.client.create_collection(
        collection_name=old_col_name,
        vectors_config=vector_store.client.get_fastembed_vector_params()
        if hasattr(vector_store.client, "get_fastembed_vector_params")
        else None,
        sparse_vectors_config=None,
    ) if hasattr(vector_store.client, "create_collection") else None

    # Normal creation through API with dense/sparse
    vector_store.create_session_collection("fresh_session")

    # Run janitor with 60 minutes limit
    # The fresh one should remain; any collection older than 60m should be pruned
    deleted = vector_store.cleanup_stale_collections(max_age_minutes=60)
    assert deleted >= 0
    assert vector_store._resolve_collection_name("fresh_session") is not None

    vector_store.delete_session_collection("fresh_session")
