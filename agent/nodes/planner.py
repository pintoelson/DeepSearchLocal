"""Planner node: Decomposes research topic into structured outline and initial search queries."""

from __future__ import annotations

import logging
from typing import Any

from agent.llm import BaseAgentLLM, StructuredOutputParser
from agent.prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt
from agent.state import ResearchState

logger = logging.getLogger(__name__)


async def planner_node(state: ResearchState, llm: BaseAgentLLM) -> dict[str, Any]:
    """Execute the planning superstep.

    Decomposes the research topic into:
    1. A multi-section outline.
    2. An initial set of orthogonal web search queries.
    """
    logger.info("Executing planner_node for topic: %s", state.research_topic)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": build_planner_user_prompt(state.research_topic)},
    ]

    try:
        raw_output = await llm.agenerate(messages, temperature=0.2, max_tokens=1000)
        parsed = StructuredOutputParser.extract_json(raw_output)

        outline = parsed.get("outline", [])
        search_queries = parsed.get("search_queries", [])

        if not isinstance(outline, list) or not outline:
            raise ValueError("Parsed outline is empty or not a list")
        if not isinstance(search_queries, list) or not search_queries:
            raise ValueError("Parsed search_queries is empty or not a list")

    except Exception as e:  # noqa: BLE001
        logger.warning("Planner LLM generation or parsing failed (%s). Using fallback plan.", e)
        # Resilient fallback decomposition
        topic = state.research_topic.strip()
        outline = [
            f"1. Executive Summary & Core Concepts of {topic}",
            f"2. Architectural & Technical Deep Dive into {topic}",
            f"3. Empirical Benchmarks & Performance Analysis of {topic}",
            f"4. Industry Applications, Trade-offs & Future Outlook of {topic}",
        ]
        search_queries = [
            f"{topic} overview architecture",
            f"{topic} benchmarks performance comparison",
            f"{topic} technical analysis documentation",
        ]

    logger.info(
        "Planner generated %d outline sections and %d initial queries",
        len(outline),
        len(search_queries),
    )

    return {
        "outline": outline,
        "search_queries": search_queries,
    }
