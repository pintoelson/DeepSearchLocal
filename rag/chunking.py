"""Deterministic recursive character chunking with Markdown heading metadata preservation."""

from __future__ import annotations

import re
from typing import Any, ClassVar

from rag.models import DocumentChunk


class RecursiveMarkdownChunker:
    """Chunks Markdown text recursively while preserving heading hierarchy and metadata.

    Features:
    - Tracks Markdown headings (# through ######) to maintain hierarchical breadcrumbs.
    - Measures exact token lengths using the bge-small tokenizer.
    - Splits text recursively using semantic Markdown boundaries:
      ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""].
    - Applies sliding-window overlap between contiguous chunks.
    - Prepends heading breadcrumb context to chunk text for dense/sparse embedding.
    """

    DEFAULT_SEPARATORS: ClassVar[list[str]] = [
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        tokenizer_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        """Initialize chunker with token limits and tokenizer."""
        if overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be less than max_tokens ({max_tokens})"
            )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

    @property
    def tokenizer(self) -> Any:
        """Lazy load tokenizer on first use."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except Exception:  # noqa: BLE001
                # Fallback to None if transformers/network is unavailable
                self._tokenizer = None
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using exact tokenizer with character fallback."""
        if not text:
            return 0
        tok = self.tokenizer
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False))
        # Fallback heuristic: ~4 characters per token
        return max(1, len(text) // 4)

    def chunk_document(
        self,
        content: str,
        source_url: str,
        page_title: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Split a Markdown document into structured DocumentChunks with heading breadcrumbs.

        Args:
            content: Raw Markdown content of the web page.
            source_url: Canonical URL of the source page.
            page_title: Page title extracted during scraping.
            extra_metadata: Optional metadata dictionary attached to each chunk.

        Returns:
            List of DocumentChunk instances ready for embedding and indexing.
        """
        if not content or not content.strip():
            return []

        # 1. Parse document into sections by heading
        sections = self._parse_markdown_sections(content)

        # 2. Chunk each section recursively
        chunks: list[DocumentChunk] = []
        global_chunk_idx = 0

        for heading_path, section_text in sections:
            if not section_text.strip():
                continue

            section_chunks = self._chunk_text(section_text)

            for chunk_raw in section_chunks:
                clean_raw = chunk_raw.strip()
                if not clean_raw:
                    continue

                tok_count = self.count_tokens(clean_raw)
                doc_chunk = DocumentChunk.create(
                    source_url=source_url,
                    page_title=page_title,
                    heading_path=heading_path,
                    raw_content=clean_raw,
                    token_count=tok_count,
                    chunk_index=global_chunk_idx,
                    metadata=extra_metadata or {},
                )
                chunks.append(doc_chunk)
                global_chunk_idx += 1

        return chunks

    def _parse_markdown_sections(self, markdown_text: str) -> list[tuple[str, str]]:
        """Split markdown into sections based on headings (#, ##, ###, etc.) and track heading breadcrumbs."""
        lines = markdown_text.splitlines()
        sections: list[tuple[str, str]] = []

        # Heading hierarchy stack: list of (level, heading_text)
        heading_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []

        heading_regex = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = heading_regex.match(line.strip())
            if match:
                # If we have accumulated text for previous heading, save it
                if current_lines:
                    text_block = "\n".join(current_lines).strip()
                    if text_block:
                        breadcrumb = " > ".join(h for _, h in heading_stack)
                        sections.append((breadcrumb, text_block))
                    current_lines = []

                level = len(match.group(1))
                heading_title = match.group(2).strip()

                # Pop headings of equal or deeper level from stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                heading_stack.append((level, heading_title))
            else:
                current_lines.append(line)

        # Append trailing lines
        if current_lines:
            text_block = "\n".join(current_lines).strip()
            if text_block:
                breadcrumb = " > ".join(h for _, h in heading_stack)
                sections.append((breadcrumb, text_block))

        # If document had no headings at all
        if not sections and current_lines:
            sections.append(("", "\n".join(current_lines).strip()))

        return sections

    def _chunk_text(self, text: str) -> list[str]:
        """Recursively split text into segments within max_tokens with overlap."""
        tokens = self.count_tokens(text)
        if tokens <= self.max_tokens:
            return [text]

        # Recursive splitting
        splits = self._split_recursive(text, self.DEFAULT_SEPARATORS)

        # Merge splits into chunks respecting max_tokens and overlap_tokens
        chunks = self._merge_splits(splits)
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Split text recursively using a hierarchy of separators."""
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        for s in splits:
            if not s:
                continue
            if self.count_tokens(s) > self.max_tokens and new_separators:
                sub_splits = self._split_recursive(s, new_separators)
                final_chunks.extend(sub_splits)
            else:
                final_chunks.append(s)

        return final_chunks

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge text segments up to max_tokens with overlap_tokens."""
        merged_chunks: list[str] = []
        current_chunk_splits: list[str] = []
        current_tokens = 0

        for split in splits:
            split_tokens = self.count_tokens(split)

            if current_tokens + split_tokens > self.max_tokens:
                if current_chunk_splits:
                    merged_chunks.append(" ".join(current_chunk_splits).strip())

                    # Retain trailing splits that fit inside overlap_tokens
                    overlap_splits: list[str] = []
                    overlap_count = 0
                    for prev_split in reversed(current_chunk_splits):
                        t_count = self.count_tokens(prev_split)
                        if overlap_count + t_count <= self.overlap_tokens:
                            overlap_splits.insert(0, prev_split)
                            overlap_count += t_count
                        else:
                            break

                    current_chunk_splits = overlap_splits + [split]
                    current_tokens = sum(self.count_tokens(s) for s in current_chunk_splits)
                else:
                    # Single split exceeds max_tokens; append directly
                    merged_chunks.append(split)
                    current_chunk_splits = []
                    current_tokens = 0
            else:
                current_chunk_splits.append(split)
                current_tokens += split_tokens

        if current_chunk_splits:
            merged_chunks.append(" ".join(current_chunk_splits).strip())

        return merged_chunks
