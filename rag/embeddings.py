"""CPU-offloaded dense and sparse embedding inference using FastEmbed (ONNX Runtime)."""

from __future__ import annotations

import logging

from fastembed import SparseTextEmbedding, TextEmbedding

logger = logging.getLogger(__name__)


class CPUEmbeddingPipeline:
    """Generates dense (bge-small-en-v1.5) and sparse (Qdrant/bm25) vectors strictly on CPU via ONNX Runtime."""

    DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
    DENSE_DIMENSION = 384

    def __init__(
        self,
        dense_model_name: str = DEFAULT_DENSE_MODEL,
        sparse_model_name: str = DEFAULT_SPARSE_MODEL,
        threads: int | None = None,
    ) -> None:
        """Initialize pipeline with model identifiers."""
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.threads = threads
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None

    @property
    def dense_model(self) -> TextEmbedding:
        """Lazy load dense TextEmbedding model on CPU."""
        if self._dense_model is None:
            logger.info("Initializing CPU dense embedder: %s", self.dense_model_name)
            self._dense_model = TextEmbedding(
                model_name=self.dense_model_name,
                threads=self.threads,
            )
        return self._dense_model

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        """Lazy load sparse BM25 SparseTextEmbedding model on CPU."""
        if self._sparse_model is None:
            logger.info("Initializing CPU sparse BM25 embedder: %s", self.sparse_model_name)
            self._sparse_model = SparseTextEmbedding(
                model_name=self.sparse_model_name,
                threads=self.threads,
            )
        return self._sparse_model

    def embed_chunks_dense(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Compute 384-dimensional dense embeddings for a batch of text chunks.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for ONNX forward passes.

        Returns:
            List of 384-dimensional float vectors.
        """
        if not texts:
            return []
        embeddings_iter = self.dense_model.embed(texts, batch_size=batch_size)
        return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in embeddings_iter]

    def embed_chunks_sparse(
        self, texts: list[str], batch_size: int = 32
    ) -> list[dict[str, list[int | float]]]:
        """Compute sparse BM25 vectors for a batch of text chunks.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for sparse inference.

        Returns:
            List of dictionaries formatted for Qdrant sparse vectors:
            [{"indices": List[int], "values": List[float]}, ...]
        """
        if not texts:
            return []
        sparse_iter = self.sparse_model.embed(texts, batch_size=batch_size)
        results: list[dict[str, list[int | float]]] = []
        for s_vec in sparse_iter:
            results.append(
                {
                    "indices": s_vec.indices.tolist()
                    if hasattr(s_vec.indices, "tolist")
                    else list(s_vec.indices),
                    "values": s_vec.values.tolist()
                    if hasattr(s_vec.values, "tolist")
                    else list(s_vec.values),
                }
            )
        return results

    def embed_query_dense(self, query: str) -> list[float]:
        """Compute dense vector for a single search query."""
        if not query or not query.strip():
            return [0.0] * self.DENSE_DIMENSION
        embeddings = list(self.dense_model.query_embed(query))
        if not embeddings:
            return [0.0] * self.DENSE_DIMENSION
        vec = embeddings[0]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def embed_query_sparse(self, query: str) -> dict[str, list[int | float]]:
        """Compute sparse BM25 vector for a single search query."""
        if not query or not query.strip():
            return {"indices": [], "values": []}
        sparse_iter = list(self.sparse_model.query_embed(query))
        if not sparse_iter:
            return {"indices": [], "values": []}
        s_vec = sparse_iter[0]
        return {
            "indices": s_vec.indices.tolist()
            if hasattr(s_vec.indices, "tolist")
            else list(s_vec.indices),
            "values": s_vec.values.tolist()
            if hasattr(s_vec.values, "tolist")
            else list(s_vec.values),
        }
