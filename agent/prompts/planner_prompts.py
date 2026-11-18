"""Prompt templates for topic decomposition and query planning."""

from __future__ import annotations

PLANNER_SYSTEM_PROMPT = """You are an expert research planner and query strategist for an autonomous deep research platform.
Your task is to decompose a given research topic into:
1. A comprehensive, logical outline (4 to 6 numbered section titles) covering the topic from fundamental principles to specific technical/empirical evidence.
2. An initial pool of 3 to 5 targeted, orthogonal web search queries designed to find concrete benchmarks, technical documentation, or factual data.

Format your entire response as a valid JSON object matching this schema:
{
  "outline": [
    "1. Overview & Architecture",
    "2. Performance & Power Efficiency",
    "3. Workload Benchmarks & Empirical Findings",
    "4. Comparative Trade-offs & Limitations"
  ],
  "search_queries": [
    "AMD Ryzen 7000 AVX-512 power consumption benchmarks",
    "Intel 13th Gen Raptor Lake peak power draw AVX2 AVX-512",
    "AnandTech Ryzen 7950X power analysis watts"
  ]
}

DO NOT include conversational prose before or after the JSON.
"""


def build_planner_user_prompt(research_topic: str) -> str:
    """Format user prompt for the planner node."""
    return f"""Please decompose the following research topic into a structured outline and initial search queries:

Topic: {research_topic}

Return strictly the JSON object.
"""
