"""Prompt templates for reflection, gap auditing, and follow-up query formulation."""

from __future__ import annotations

REFLECT_SYSTEM_PROMPT = """You are a critical scientific research auditor and reflection engine.
Your task is to audit the currently drafted sections of a research report against the research topic and target outline.

EVALUATION CRITERIA:
1. Coverage: Does each section answer the technical questions implied by its outline title?
2. Factual Grounding: Are there vague generalizations, missing quantitative data, or speculative assertions that lack citations?
3. Contradictions & Gaps: Are there conflicting figures or unresolved questions that require additional web search?

OUTPUT FORMAT:
Return strictly a valid JSON object matching this schema:
{
  "has_gaps": true,
  "sections_to_revise": [
    "2. Performance & Power Efficiency"
  ],
  "unresolved_gaps": [
    "Missing exact power draw numbers under AVX-512 workload for Intel Core i9-13900K"
  ],
  "follow_up_queries": [
    "Intel Core i9-13900K power consumption AVX-512 watts measurement"
  ]
}

If all sections are thoroughly grounded, complete, and adequately cited, set "has_gaps" to false, "sections_to_revise" to [], "unresolved_gaps" to [], and "follow_up_queries" to [].
DO NOT include any Markdown formatting or prose outside the JSON object.
"""


def build_reflect_user_prompt(
    research_topic: str,
    outline: list[str],
    sections: dict[str, str],
) -> str:
    """Format user prompt for the reflection and audit node."""
    outline_text = "\n".join(f"- {s}" for s in outline)

    drafts_text = []
    for title, draft in sections.items():
        snippet = draft[:800] + ("..." if len(draft) > 800 else "")
        drafts_text.append(f"### {title}\n{snippet}\n")

    combined_drafts = "\n".join(drafts_text) if drafts_text else "No sections drafted yet."

    return f"""Research Topic: {research_topic}

Expected Outline:
{outline_text}

Draft Summaries:
{combined_drafts}

Please audit the drafts against the research topic and outline. Determine whether significant information gaps exist and formulate targeted follow-up queries if necessary.
"""
