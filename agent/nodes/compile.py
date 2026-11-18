"""Compiler node: Two-pass citation normalization, bibliography compilation, and ephemeral teardown."""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.state import ResearchState
from rag.vector_store import EphemeralVectorStore

logger = logging.getLogger(__name__)

# Matches [^chunk_id] syntax
CITATION_TAG_PATTERN = re.compile(r"\[\^([a-zA-Z0-9_-]+)\]")


def compiler_node(
    state: ResearchState,
    vector_store: EphemeralVectorStore | None = None,
) -> dict[str, Any]:
    """Execute the final compilation superstep.

    Responsibilities:
    1. Consolidate section drafts in logical outline order.
    2. Pass 2 Citation Resolution:
       - Normalizes temporary chunk hash tags ([^chunk_hash]) into sequential numeric markers ([1], [2]).
       - Deduplicates citations across sections.
       - Strips unverified/hallucinated orphan tags and logs them.
    3. Generates unified Markdown bibliography with titles and source URLs.
    4. Populates final_report and citation_map for UI/SSE streaming.
    5. Tears down ephemeral Qdrant collection unless retain_collection is True.
    """
    logger.info("Executing compiler_node for session=%s", state.session_id)

    # 1. Build lookup table of recognized chunks
    chunk_lookup: dict[str, dict[str, Any]] = {}
    for chunk in state.evidence_chunks:
        cid = str(chunk.get("chunk_id", "")).strip()
        if cid:
            chunk_lookup[cid] = chunk
            # Also index 8-char prefix in case model truncated hash
            if len(cid) >= 8:
                chunk_lookup[cid[:8]] = chunk

    logger.info(
        "Compiler initialized with %d verified context chunks for citation mapping",
        len(chunk_lookup),
    )

    # 2. Sequential Citation Normalization & Deduplication
    canonical_chunk_to_num: dict[str, int] = {}
    ordered_references: list[dict[str, Any]] = []
    orphan_tags_detected: list[str] = []
    citation_counter = 1

    def normalize_match(match: re.Match[str]) -> str:
        nonlocal citation_counter
        tag = match.group(1).strip()

        if tag in chunk_lookup:
            matched_chunk = chunk_lookup[tag]
            canonical_id = str(matched_chunk.get("chunk_id", tag))

            if canonical_id not in canonical_chunk_to_num:
                canonical_chunk_to_num[canonical_id] = citation_counter
                ordered_references.append({
                    "citation_num": citation_counter,
                    "chunk": matched_chunk,
                })
                citation_counter += 1

            num = canonical_chunk_to_num[canonical_id]
            return f"[{num}]"
        else:
            # Orphan / hallucinated citation tag: strip from text per design alignment
            orphan_tags_detected.append(tag)
            return ""

    # Determine section ordering
    ordered_sections: list[tuple[str, str]] = []
    if state.outline:
        for title in state.outline:
            draft = state.sections.get(title, "")
            if draft:
                ordered_sections.append((title, draft))
    # Add any sections present in state.sections that were not in outline
    existing_titles = {t for t, _ in ordered_sections}
    for title, draft in state.sections.items():
        if title not in existing_titles and draft:
            ordered_sections.append((title, draft))

    # Normalize citations in each section
    compiled_sections: list[str] = []
    for title, draft in ordered_sections:
        normalized_draft = CITATION_TAG_PATTERN.sub(normalize_match, draft)
        # Ensure section begins with heading if draft doesn't already contain it
        clean_draft = normalized_draft.strip()
        if not clean_draft.startswith("#"):
            section_md = f"## {title}\n\n{clean_draft}"
        else:
            section_md = clean_draft
        compiled_sections.append(section_md)

    if orphan_tags_detected:
        logger.warning(
            "Compiler stripped %d orphan citation tags not found in evidence context: %s",
            len(orphan_tags_detected),
            orphan_tags_detected[:10],
        )

    # 3. Construct Unified Bibliography
    bibliography_lines = ["## References\n"]
    citation_map: dict[str, dict[str, str]] = {}

    if ordered_references:
        for ref in ordered_references:
            num = ref["citation_num"]
            chunk = ref["chunk"]
            url = chunk.get("source_url", "")
            title = chunk.get("page_title", "") or "Web Source"
            snippet = chunk.get("raw_content", "")[:200].replace("\n", " ").strip()
            cid = chunk.get("chunk_id", "")

            if url:
                entry = f"[{num}] **{title}** - [{url}]({url})"
            else:
                entry = f"[{num}] **{title}**"
            bibliography_lines.append(entry)

            citation_map[str(num)] = {
                "number": str(num),
                "title": title,
                "url": url,
                "chunk_id": cid,
                "snippet": snippet,
            }
    else:
        bibliography_lines.append("*(No external citations cited in this report)*")

    # 4. Assemble Final Report
    report_title = state.research_topic.strip()
    report_header = f"# {report_title}\n\n"
    report_body = "\n\n".join(compiled_sections)
    report_footer = "\n\n" + "\n".join(bibliography_lines)

    final_report = report_header + report_body + report_footer

    # 5. Clean up ephemeral Qdrant collection
    if vector_store and not state.retain_collection:
        try:
            logger.info("Purging ephemeral Qdrant collection for session %s", state.session_id)
            vector_store.delete_session_collection(state.session_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to delete ephemeral session collection %s: %s", state.session_id, e)

    logger.info(
        "Compiler completed successfully. Report length=%d chars, unique citations=%d",
        len(final_report),
        len(ordered_references),
    )

    return {
        "final_report": final_report,
        "citation_map": citation_map,
    }
