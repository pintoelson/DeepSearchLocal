"""Data models for the DeepSearch-Local Hybrid RAG subsystem."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunked segment of text from a scraped document."""

    chunk_id: str = Field(
        description="Deterministic hash identifying this chunk (used for citation markers)."
    )
    source_url: str = Field(description="URL of the source web page.")
    page_title: str = Field(default="", description="Title of the source web page.")
    heading_path: str = Field(
        default="", description="Hierarchical breadcrumb path (e.g. 'Overview > Architecture')."
    )
    raw_content: str = Field(description="Unmodified chunk text for exact citational quotation.")
    embedding_text: str = Field(
        description="Text with prepended heading context used for dense & sparse embeddings."
    )
    token_count: int = Field(
        default=0, description="Measured token length using bge-small tokenizer."
    )
    chunk_index: int = Field(
        default=0, description="Sequential index of the chunk within the source page."
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata.")

    @classmethod
    def create(
        cls,
        source_url: str,
        page_title: str,
        heading_path: str,
        raw_content: str,
        token_count: int,
        chunk_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        """Create a DocumentChunk with deterministic hash ID and context prefixing."""
        # Clean heading path and raw content
        heading_clean = heading_path.strip()
        raw_clean = raw_content.strip()

        # Context prefixing: Prepend heading breadcrumbs to embedding text if available
        if heading_clean:
            embedding_text = f"[Context: {heading_clean}]\n\n{raw_clean}"
        else:
            embedding_text = raw_clean

        # Deterministic chunk ID hash
        hash_seed = f"{source_url}|{chunk_index}|{raw_clean[:64]}"
        chunk_id = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:12]

        return cls(
            chunk_id=chunk_id,
            source_url=source_url,
            page_title=page_title,
            heading_path=heading_clean,
            raw_content=raw_clean,
            embedding_text=embedding_text,
            token_count=token_count,
            chunk_index=chunk_index,
            metadata=metadata or {},
        )


class ScoredChunk(BaseModel):
    """DocumentChunk enriched with retrieval and reranking scores."""

    chunk: DocumentChunk
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion score from Qdrant.")
    reranker_score: float | None = Field(
        default=None, description="Normalized relevance score [0, 1] from the cross-encoder."
    )
