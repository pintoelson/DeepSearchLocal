"""Ingestion node: Splits Markdown documents and indexes embeddings into ephemeral Qdrant collection."""

from __future__ import annotations

import logging
from typing import Any

from agent.state import ResearchState
from rag.chunking import RecursiveMarkdownChunker
from rag.models import DocumentChunk
from rag.vector_store import EphemeralVectorStore

logger = logging.getLogger(__name__)


def ingestion_node(
    state: ResearchState,
    vector_store: EphemeralVectorStore,
    chunker: RecursiveMarkdownChunker | None = None,
) -> dict[str, Any]:
    """Execute ingestion superstep.

    Features:
    - Guarantees ephemeral Qdrant session collection exists.
    - Deterministically chunks scraped documents using heading preservation.
    - CPU-offloaded dense (bge-small) and sparse (BM25) embedding indexing.
    - Returns serialized evidence_chunks update and clears raw_documents.
    """
    logger.info(
        "Executing ingestion_node for session=%s with %d raw documents",
        state.session_id,
        len(state.raw_documents),
    )

    vector_store.create_session_collection(state.session_id)

    if not state.raw_documents:
        logger.info("No raw documents to ingest in this superstep.")
        return {
            "evidence_chunks": [],
            "raw_documents": [],
        }

    chunker_instance = chunker or RecursiveMarkdownChunker(max_tokens=512, overlap_tokens=64)
    all_chunks: list[DocumentChunk] = []

    for doc in state.raw_documents:
        content = doc.get("content", "")
        url = doc.get("url", "")
        title = doc.get("title", "")
        if not content:
            continue

        doc_chunks = chunker_instance.chunk_document(
            content=content,
            source_url=url,
            page_title=title,
        )
        all_chunks.extend(doc_chunks)

    logger.info("Generated %d chunks from %d raw documents", len(all_chunks), len(state.raw_documents))

    if all_chunks:
        upserted_count = vector_store.add_documents(session_id=state.session_id, chunks=all_chunks)
        logger.info("Successfully upserted %d points to Qdrant collection for session %s", upserted_count, state.session_id)

    serialized_chunks = [c.model_dump() for c in all_chunks]

    return {
        "evidence_chunks": serialized_chunks,
        "raw_documents": [],
    }
