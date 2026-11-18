"""State definitions for the DeepSearch-Local LangGraph agent."""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any

from pydantic import BaseModel, Field


def merge_url_sets(existing: set[str], update: set[str]) -> set[str]:
    """Channel reducer for merging scraped URL sets without inspect.signature issues."""
    return existing.union(update)


class ResearchState(BaseModel):
    """Pydantic state machine schema for multi-step cyclic deep research."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    research_topic: str
    outline: list[str] = Field(default_factory=list)
    search_queries: Annotated[list[str], operator.add] = Field(default_factory=list)
    scraped_urls: Annotated[set[str], merge_url_sets] = Field(default_factory=set)
    evidence_chunks: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    raw_documents: list[dict[str, Any]] = Field(default_factory=list)
    sections: dict[str, str] = Field(default_factory=dict)
    sections_to_revise: list[str] = Field(default_factory=list)
    unresolved_gaps: list[str] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 2
    final_report: str | None = None
    citation_map: dict[str, dict[str, str]] = Field(default_factory=dict)
    retain_collection: bool = False
