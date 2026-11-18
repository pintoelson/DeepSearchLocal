"""Cross-encoder re-ranking wrapper for candidate passages using FastEmbed (ONNX Runtime)."""

from __future__ import annotations

import logging
import math

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.models import ScoredChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ranks retrieved candidates using a lightweight cross-encoder (BAAI/bge-reranker-base) on CPU."""

    DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        threads: int | None = None,
    ) -> None:
        """Initialize cross-encoder reranker with ONNX model identifier."""
        self.model_name = model_name
        self.threads = threads
        self._reranker: TextCrossEncoder | None = None

    @property
    def reranker(self) -> TextCrossEncoder:
        """Lazy load ONNX cross-encoder model on CPU."""
        if self._reranker is None:
            logger.info("Initializing CPU Cross-Encoder Reranker: %s", self.model_name)
            self._reranker = TextCrossEncoder(
                model_name=self.model_name,
                threads=self.threads,
            )
        return self._reranker

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_n: int = 4,
    ) -> list[ScoredChunk]:
        """Score (query, doc) pairs with full cross-attention and filter down to top_n passages.

        Args:
            query: The research sub-query or section topic being synthesized.
            candidates: Top-15 fused candidate chunks from hybrid retrieval.
            top_n: Number of final high-confidence chunks to retain (default: 4).

        Returns:
            Filtered list of ScoredChunk instances sorted descending by reranker_score.
        """
        if not candidates or top_n <= 0:
            return []

        if not query or not query.strip():
            # If query is empty, return top_n based on original RRF score
            return sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:top_n]

        # Extract texts for scoring (use embedding_text to evaluate heading context)
        texts_to_score = [c.chunk.embedding_text for c in candidates]

        # Execute ONNX cross-encoder inference on CPU
        raw_scores = list(self.reranker.rerank(query, texts_to_score))

        # Assign sigmoid-normalized relevance scores to each candidate
        scored_candidates: list[ScoredChunk] = []
        for candidate, logit in zip(candidates, raw_scores):
            # Sigmoid normalization: maps [-inf, inf] logit to [0, 1] probability
            try:
                prob = 1.0 / (1.0 + math.exp(-float(logit)))
            except OverflowError:
                prob = 1.0 if logit > 0 else 0.0

            candidate.reranker_score = prob
            scored_candidates.append(candidate)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda c: c.reranker_score or 0.0, reverse=True)

        return scored_candidates[:top_n]
