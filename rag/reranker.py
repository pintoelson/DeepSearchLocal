"""Cross-encoder re-ranking wrapper for candidate passages."""

from typing import Any, Dict, List


class CrossEncoderReranker:
    """Re-ranks retrieved candidates using a lightweight cross-encoder (BAAI/bge-reranker-base)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
        """Score (query, doc) pairs and return top_n filtered passages."""
        raise NotImplementedError("Cross-encoder reranker implementation in Phase 2.")
