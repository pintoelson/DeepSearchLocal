"""Reflection node: Audits section drafts, detects factual gaps, and triggers revision loops."""

from __future__ import annotations

import logging
from typing import Any

from agent.llm import BaseAgentLLM, StructuredOutputParser
from agent.prompts.reflect_prompts import REFLECT_SYSTEM_PROMPT, build_reflect_user_prompt
from agent.state import ResearchState

logger = logging.getLogger(__name__)


async def reflection_node(state: ResearchState, llm: BaseAgentLLM) -> dict[str, Any]:
    """Execute the reflection and audit superstep.

    Evaluates drafts against outline:
    - If current iteration < max_iterations:
      Audits section drafts and identifies missing facts.
      If gaps exist and next iteration is within limit, triggers targeted revision loop.
    - If max_iterations reached:
      Halts revisions to prevent infinite cycling.
    """
    current_iter = state.iteration_count
    next_iter = current_iter + 1
    logger.info(
        "Executing reflection_node: completed iteration %d (max=%d)",
        current_iter,
        state.max_iterations,
    )

    # If maximum iterations already reached prior to this step, route directly to compiler
    if current_iter >= state.max_iterations:
        logger.info(
            "Max iterations limit reached (%d >= %d). Halting reflection cycles.",
            current_iter,
            state.max_iterations,
        )
        return {
            "iteration_count": current_iter,
            "sections_to_revise": [],
            "unresolved_gaps": state.unresolved_gaps,
            "search_queries": [],
        }

    messages = [
        {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_reflect_user_prompt(
                research_topic=state.research_topic,
                outline=state.outline,
                sections=state.sections,
            ),
        },
    ]

    try:
        raw_output = await llm.agenerate(messages, temperature=0.2, max_tokens=1000)
        parsed = StructuredOutputParser.extract_json(raw_output)

        has_gaps = bool(parsed.get("has_gaps", False))
        raw_sections = parsed.get("sections_to_revise", []) if has_gaps else []
        unresolved_gaps = parsed.get("unresolved_gaps", []) if has_gaps else []
        follow_up_queries = parsed.get("follow_up_queries", []) if has_gaps else []

        if not isinstance(raw_sections, list):
            raw_sections = []
        if not isinstance(unresolved_gaps, list):
            unresolved_gaps = []
        if not isinstance(follow_up_queries, list):
            follow_up_queries = []

    except Exception as e:  # noqa: BLE001
        logger.warning("Reflection parsing failed (%s). Assuming draft complete.", e)
        has_gaps = False
        raw_sections = []
        unresolved_gaps = []
        follow_up_queries = []

    # If next iteration will hit or exceed max_iterations, we cannot schedule more revisions
    if has_gaps and next_iter < state.max_iterations:
        sections_to_revise = raw_sections
    else:
        if has_gaps:
            logger.info("Gaps detected but next iteration %d reaches max_iterations %d. Finalizing draft.", next_iter, state.max_iterations)
        sections_to_revise = []

    logger.info(
        "Reflection audit result: %d sections to revise, %d follow-up queries generated (next_iter=%d)",
        len(sections_to_revise),
        len(follow_up_queries),
        next_iter,
    )

    return {
        "iteration_count": next_iter,
        "sections_to_revise": sections_to_revise,
        "unresolved_gaps": unresolved_gaps,
        "search_queries": follow_up_queries if sections_to_revise else [],
    }
