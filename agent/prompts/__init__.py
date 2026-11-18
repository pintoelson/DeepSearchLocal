"""Prompt templates for DeepSearch-Local LangGraph nodes."""

from agent.prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt
from agent.prompts.reflect_prompts import REFLECT_SYSTEM_PROMPT, build_reflect_user_prompt
from agent.prompts.synth_prompts import (
    SYNTH_REVISION_SYSTEM_PROMPT,
    SYNTH_SYSTEM_PROMPT,
    build_synth_revision_user_prompt,
    build_synth_user_prompt,
    format_evidence_passages,
)

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "REFLECT_SYSTEM_PROMPT",
    "SYNTH_REVISION_SYSTEM_PROMPT",
    "SYNTH_SYSTEM_PROMPT",
    "build_planner_user_prompt",
    "build_reflect_user_prompt",
    "build_synth_revision_user_prompt",
    "build_synth_user_prompt",
    "format_evidence_passages",
]
