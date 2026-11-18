"""Hybrid retrieval combining dense vector embeddings with BM25 sparse search and server-side RRF."""

from __future__ import annotations

import logging
from typing import Literal

from qdrant_client import models

from rag.models import DocumentChunk, ScoredChunk
from rag.vector_store import EphemeralVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuses dense semantic search (bge-small-en-v1.5) and BM25 using Qdrant server-side RRF."""

    def __init__(
        self,
        vector_store: EphemeralVectorStore,
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid retriever with vector store instance and RRF smoothing constant."""
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        session_id: str,
        query: str,
        top_k: int = 15,
        prefetch_limit: int = 50,
        mode: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid",
    ) -> list[ScoredChunk]:
        """Retrieve and rank top candidates via server-side prefetch, dense-only, or sparse-only search.

        Args:
            session_id: Research session identifier.
            query: User search query or section topic.
            top_k: Number of unified candidates to return (default: 15).
            prefetch_limit: Candidate pool size per retrieval modality (default: 50).
            mode: Retrieval mode ('hybrid', 'dense_only', or 'sparse_only').

        Returns:
            List of ScoredChunk instances sorted by relevance rank.
        """
        if not query or not query.strip():
            return []

        collection_name = self.vector_store._resolve_collection_name(session_id)
        if not collection_name:
            logger.warning(
                "No collection found for session_id=%s. Returning empty results.", session_id
            )
            return []

        if mode == "dense_only":
            dense_query = self.vector_store.embedding_pipeline.embed_query_dense(query)
            response = self.vector_store.client.query_points(
                collection_name=collection_name,
                query=dense_query,
                using="",
                limit=top_k,
                with_payload=True,
            )
        elif mode == "sparse_only":
            sparse_query = self.vector_store.embedding_pipeline.embed_query_sparse(query)
            if not sparse_query["indices"]:
                return []
            response = self.vector_store.client.query_points(
                collection_name=collection_name,
                query=models.SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"],
                ),
                using=EphemeralVectorStore.SPARSE_VECTOR_NAME,
                limit=top_k,
                with_payload=True,
            )
        else:
            # Hybrid search with server-side RRF fusion
            dense_query = self.vector_store.embedding_pipeline.embed_query_dense(query)
            sparse_query = self.vector_store.embedding_pipeline.embed_query_sparse(query)

            prefetch_queries = [
                models.Prefetch(
                    query=dense_query,
                    using="",
                    limit=prefetch_limit,
                ),
            ]

            if sparse_query["indices"]:
                prefetch_queries.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query["indices"],
                            values=sparse_query["values"],
                        ),
                        using=EphemeralVectorStore.SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    )
                )

            response = self.vector_store.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_queries,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

        # 4. Convert Qdrant ScoredPoints to ScoredChunk models
        scored_chunks: list[ScoredChunk] = []
        for point in response.points:
            payload = point.payload or {}
            doc_chunk = DocumentChunk(
                chunk_id=str(payload.get("chunk_id", point.id)),
                source_url=str(payload.get("source_url", "")),
                page_title=str(payload.get("page_title", "")),
                heading_path=str(payload.get("heading_path", "")),
                raw_content=str(payload.get("raw_content", "")),
                embedding_text=str(payload.get("embedding_text", "")),
                token_count=int(payload.get("token_count", 0)),
                chunk_index=int(payload.get("chunk_index", 0)),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "chunk_id",
                        "source_url",
                        "page_title",
                        "heading_path",
                        "raw_content",
                        "embedding_text",
                        "token_count",
                        "chunk_index",
                    }
                },
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=doc_chunk,
                    rrf_score=float(point.score if point.score is not None else 0.0),
                )
            )

        return scored_chunks
