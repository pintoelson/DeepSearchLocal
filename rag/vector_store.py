"""Qdrant client and ephemeral session collection lifecycle management."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from qdrant_client import QdrantClient, models

from rag.embeddings import CPUEmbeddingPipeline
from rag.models import DocumentChunk

logger = logging.getLogger(__name__)


class EphemeralVectorStore:
    """Manages isolated, on-the-fly Qdrant vector collections per research session.

    Supports:
    - Dual-Mode: Containerized Qdrant (localhost:6333) or in-process (:memory:) mode.
    - Ephemeral session collection provisioning with dense (384d Cosine) and sparse (BM25) vector schemas.
    - Batch embedding and upsert of DocumentChunks.
    - Synchronous and context manager lifecycle deletion.
    - Automated cleanup of stale/orphaned session collections.
    """

    COLLECTION_PREFIX = "session"
    DEFAULT_DENSE_DIMENSION = 384
    SPARSE_VECTOR_NAME = "bm25"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        use_memory: bool = False,
        embedding_pipeline: CPUEmbeddingPipeline | None = None,
    ) -> None:
        """Initialize EphemeralVectorStore in either network or in-memory mode."""
        self.host = host
        self.port = port
        self.use_memory = use_memory
        self.embedding_pipeline = embedding_pipeline or CPUEmbeddingPipeline()

        if self.use_memory:
            logger.info("Initializing EphemeralVectorStore in :memory: mode.")
            self.client = QdrantClient(location=":memory:")
        else:
            logger.info("Connecting EphemeralVectorStore to Qdrant at %s:%s", host, port)
            try:
                self.client = QdrantClient(host=host, port=port, timeout=5.0)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to connect to Qdrant at %s:%s (%s). Falling back to :memory: mode.",
                    host,
                    port,
                    e,
                )
                self.client = QdrantClient(location=":memory:")
                self.use_memory = True

    def _resolve_collection_name(self, session_id: str) -> str | None:
        """Find an existing collection name matching the session_id."""
        collections = [c.name for c in self.client.get_collections().collections]
        for name in collections:
            if name == f"{self.COLLECTION_PREFIX}_{session_id}" or name.endswith(f"_{session_id}"):
                return name
        return None

    def create_session_collection(self, session_id: str) -> str:
        """Create a dedicated ephemeral collection with dense and sparse vector schemas.

        Collection naming format: session_{timestamp}_{session_id}
        """
        existing = self._resolve_collection_name(session_id)
        if existing:
            logger.info("Session collection already exists: %s", existing)
            return existing

        timestamp = int(time.time())
        collection_name = f"{self.COLLECTION_PREFIX}_{timestamp}_{session_id}"

        logger.info("Creating ephemeral Qdrant collection: %s", collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.DEFAULT_DENSE_DIMENSION,
                distance=models.Distance.COSINE,
            ),
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        return collection_name

    def delete_session_collection(self, session_id: str) -> bool:
        """Clean up ephemeral collection after research completion or session timeout."""
        collection_name = self._resolve_collection_name(session_id)
        if not collection_name:
            logger.warning("No collection found for session_id=%s to delete.", session_id)
            return False

        logger.info("Deleting ephemeral Qdrant collection: %s", collection_name)
        result = self.client.delete_collection(collection_name=collection_name)
        return bool(result)

    def add_documents(
        self,
        session_id: str,
        chunks: list[DocumentChunk],
        batch_size: int = 32,
    ) -> int:
        """Embed and upsert document chunks into the session's ephemeral collection.

        Args:
            session_id: Research session identifier.
            chunks: List of DocumentChunk instances.
            batch_size: Embedding and upsert batch size.

        Returns:
            Number of points successfully upserted.
        """
        if not chunks:
            return 0

        collection_name = self._resolve_collection_name(session_id)
        if not collection_name:
            collection_name = self.create_session_collection(session_id)

        total_upserted = 0
        texts_to_embed = [c.embedding_text for c in chunks]

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = texts_to_embed[i : i + batch_size]

            # Generate CPU dense vectors
            dense_vectors = self.embedding_pipeline.embed_chunks_dense(batch_texts)

            # Generate CPU sparse vectors
            sparse_vectors = self.embedding_pipeline.embed_chunks_sparse(batch_texts)

            points: list[models.PointStruct] = []
            for chunk, dense_vec, sparse_vec in zip(batch_chunks, dense_vectors, sparse_vectors):
                # Derive deterministic UUID for Qdrant point
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{session_id}:{chunk.chunk_id}"))

                # Package payload
                payload: dict[str, Any] = {
                    "chunk_id": chunk.chunk_id,
                    "source_url": chunk.source_url,
                    "page_title": chunk.page_title,
                    "heading_path": chunk.heading_path,
                    "raw_content": chunk.raw_content,
                    "embedding_text": chunk.embedding_text,
                    "token_count": chunk.token_count,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                }

                # Construct PointStruct with unnamed dense and named sparse vector
                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "": dense_vec,
                        self.SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vec["indices"],
                            values=sparse_vec["values"],
                        ),
                    },
                    payload=payload,
                )
                points.append(point)

            self.client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)

        return total_upserted

    def count_documents(self, session_id: str) -> int:
        """Count total documents indexed in this session collection."""
        collection_name = self._resolve_collection_name(session_id)
        if not collection_name:
            return 0
        return self.client.count(collection_name=collection_name).count

    def cleanup_stale_collections(self, max_age_minutes: int = 60) -> int:
        """Purge orphaned or timed-out session collections older than max_age_minutes.

        Returns:
            Number of collections deleted.
        """
        now = time.time()
        max_age_seconds = max_age_minutes * 60
        deleted_count = 0

        collections = [c.name for c in self.client.get_collections().collections]
        pattern = re.compile(rf"^{self.COLLECTION_PREFIX}_(\d+)_(.+)$")

        for name in collections:
            match = pattern.match(name)
            if match:
                created_timestamp = int(match.group(1))
                age = now - created_timestamp
                if age > max_age_seconds:
                    logger.info("Pruning stale session collection: %s (age: %ds)", name, int(age))
                    self.client.delete_collection(name)
                    deleted_count += 1

        return deleted_count

    @contextmanager
    def session(self, session_id: str) -> Generator[str, None, None]:
        """Context manager to guarantee creation and teardown of an ephemeral session collection."""
        col_name = self.create_session_collection(session_id)
        try:
            yield col_name
        finally:
            self.delete_session_collection(session_id)
