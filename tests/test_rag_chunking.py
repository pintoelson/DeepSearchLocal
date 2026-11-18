"""Unit tests for rag.chunking (RecursiveMarkdownChunker)."""

import pytest

from rag.chunking import RecursiveMarkdownChunker


def test_chunker_empty_and_whitespace():
    """Verify chunker gracefully handles empty or whitespace-only inputs."""
    chunker = RecursiveMarkdownChunker()
    assert chunker.chunk_document("", "https://example.com") == []
    assert chunker.chunk_document("   \n\n  ", "https://example.com") == []


def test_chunker_heading_breadcrumb_tracking():
    """Verify heading stack tracks hierarchical breadcrumbs properly."""
    content = """# DeepSearch Architecture
This is top-level introduction.

## Memory Specifications
The RTX 3050 has 6GB VRAM.

### KV Cache Sizing
4096 context tokens consume approximately 1GB VRAM.

## Processor Core
The CPU handles RAG and embedding inference.
"""
    chunker = RecursiveMarkdownChunker(max_tokens=512)
    chunks = chunker.chunk_document(
        content, "https://example.com/arch", page_title="DeepSearch Arch"
    )

    assert len(chunks) == 4

    # Chunk 0: Top-level
    assert chunks[0].heading_path == "DeepSearch Architecture"
    assert "top-level introduction" in chunks[0].raw_content
    assert chunks[0].embedding_text.startswith("[Context: DeepSearch Architecture]\n\n")

    # Chunk 1: Sub-heading
    assert chunks[1].heading_path == "DeepSearch Architecture > Memory Specifications"
    assert "RTX 3050 has 6GB VRAM" in chunks[1].raw_content
    assert chunks[1].embedding_text.startswith(
        "[Context: DeepSearch Architecture > Memory Specifications]\n\n"
    )

    # Chunk 2: Sub-sub-heading
    assert (
        chunks[2].heading_path
        == "DeepSearch Architecture > Memory Specifications > KV Cache Sizing"
    )
    assert "4096 context tokens" in chunks[2].raw_content
    assert chunks[2].embedding_text.startswith(
        "[Context: DeepSearch Architecture > Memory Specifications > KV Cache Sizing]\n\n"
    )

    # Chunk 3: Sibling heading replacing Memory Specifications
    assert chunks[3].heading_path == "DeepSearch Architecture > Processor Core"
    assert "CPU handles RAG" in chunks[3].raw_content
    assert chunks[3].embedding_text.startswith(
        "[Context: DeepSearch Architecture > Processor Core]\n\n"
    )


def test_chunker_token_limit_and_recursive_splitting():
    """Verify long text blocks are split into chunks under max_tokens."""
    # Create a long text block with 20 paragraphs
    paragraphs = [
        f"Paragraph {i}: DeepSearch-Local is an autonomous research agent platform running on local consumer GPUs. "
        f"It pairs a 3B SLM with a hybrid RAG engine offloaded to CPU."
        for i in range(25)
    ]
    content = "# Performance Overview\n\n" + "\n\n".join(paragraphs)

    max_tokens = 64
    overlap_tokens = 16
    chunker = RecursiveMarkdownChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    chunks = chunker.chunk_document(content, "https://example.com/perf")

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= max_tokens + 5  # Allow tiny boundary slack
        assert chunk.heading_path == "Performance Overview"
        assert chunk.embedding_text.startswith("[Context: Performance Overview]\n\n")


def test_chunker_deterministic_chunk_ids():
    """Verify identical text and URL generates deterministic chunk IDs."""
    content = "# Test\nSome sample content."
    chunker = RecursiveMarkdownChunker()

    chunks_1 = chunker.chunk_document(content, "https://example.com/page")
    chunks_2 = chunker.chunk_document(content, "https://example.com/page")

    assert chunks_1[0].chunk_id == chunks_2[0].chunk_id

    # Different URL should yield different chunk_id
    chunks_different_url = chunker.chunk_document(content, "https://example.com/other")
    assert chunks_1[0].chunk_id != chunks_different_url[0].chunk_id


def test_chunker_invalid_overlap_raises():
    """Verify ValueError is raised if overlap_tokens >= max_tokens."""
    with pytest.raises(ValueError, match="must be less than max_tokens"):
        RecursiveMarkdownChunker(max_tokens=100, overlap_tokens=100)
