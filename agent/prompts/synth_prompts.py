"""Prompt templates for grounded section synthesis with temporary chunk citations."""

from __future__ import annotations

from typing import Any

SYNTH_SYSTEM_PROMPT = """You are an objective, rigorous scientific research synthesizer.
Your task is to write a detailed, highly technical section for a deep research report using ONLY the provided verified context passages.

CRITICAL GROUNDING & CITATION RULES:
1. Every factual assertion, benchmark number, architectural specification, or metric MUST be directly cited inline using the verbatim chunk identifier: [^chunk_id].
2. For example, if Chunk [c8a1f29b] says "The 7950X consumes 230W under full AVX-512 load", write:
   "The AMD Ryzen 7950X reaches a peak power draw of 230W during continuous AVX-512 execution [^c8a1f29b]."
3. NEVER invent or hallucinate citation identifiers. ONLY cite chunk identifiers explicitly present in the provided evidence.
4. DO NOT use sequential numbers like [1], [2]. Always use [^<chunk_id>].
5. If the evidence does not contain sufficient facts to answer a specific aspect, state the limitation objectively rather than guessing.
6. Write in comprehensive, analytical Markdown with subsections, bullet points, or tables if appropriate. Do NOT include references or bibliography at the end of the section (the compiler node handles this globally).
"""

SYNTH_REVISION_SYSTEM_PROMPT = """You are an objective scientific research synthesizer performing an iterative revision on a research draft.
Your task is to enrich and update the existing draft for a section using newly retrieved evidence passages.

CRITICAL REVISION RULES:
1. Address missing facts, unresolved questions, or unverified claims present in the initial draft.
2. Integrate facts from the newly provided evidence passages seamlessly into the text.
3. Every new or updated factual claim MUST be cited with the exact chunk identifier: [^chunk_id].
4. Retain valid existing citations from the prior draft while adding citations for new assertions.
5. Produce the complete, updated section text in Markdown.
"""


def format_evidence_passages(chunks: list[dict[str, Any]]) -> str:
    """Format candidate chunks into clear, tagged context blocks for the prompt."""
    if not chunks:
        return "No specific evidence chunks available for this section."

    formatted_blocks = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", "unknown")
        url = chunk.get("source_url", "")
        title = chunk.get("page_title", "")
        heading = chunk.get("heading_path", "")
        content = chunk.get("raw_content", "").strip()

        header = f"[Chunk {cid}]"
        if title:
            header += f" Title: {title}"
        if heading:
            header += f" | Section: {heading}"
        if url:
            header += f" | Source: {url}"

        block = f"{header}\n{content}"
        formatted_blocks.append(block)

    return "\n\n---\n\n".join(formatted_blocks)


def build_synth_user_prompt(
    section_title: str,
    research_topic: str,
    outline: list[str],
    evidence_chunks: list[dict[str, Any]],
) -> str:
    """Build synthesis prompt for an initial section draft."""
    evidence_text = format_evidence_passages(evidence_chunks)
    outline_context = "\n".join(f"- {item}" for item in outline)

    return f"""Research Topic: {research_topic}

Full Report Outline Context:
{outline_context}

Target Section to Write:
## {section_title}

Verified Evidence Passages:
{evidence_text}

Instructions:
Draft the comprehensive technical content for section "## {section_title}". Ground every claim with [^chunk_id] tags from the evidence above. Output only the Markdown text for this section.
"""


def build_synth_revision_user_prompt(
    section_title: str,
    research_topic: str,
    current_draft: str,
    unresolved_gaps: list[str],
    new_evidence_chunks: list[dict[str, Any]],
) -> str:
    """Build synthesis prompt for revising an existing section with new evidence."""
    evidence_text = format_evidence_passages(new_evidence_chunks)
    gaps_text = "\n".join(f"- {gap}" for gap in unresolved_gaps) if unresolved_gaps else "Enhance factual grounding and precision."

    return f"""Research Topic: {research_topic}
Target Section to Revise: ## {section_title}

Identified Information Gaps to Address:
{gaps_text}

Current Section Draft:
{current_draft}

Newly Retrieved Evidence Passages:
{evidence_text}

Instructions:
Revise and expand the section draft to resolve the identified information gaps using the new evidence. Ensure all new assertions are cited with [^chunk_id]. Return the full revised section Markdown.
"""
