"""State definitions for the DeepSearch-Local LangGraph agent."""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    """Pydantic state machine schema for multi-step deep research."""

    research_topic: str
    outline: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    scraped_urls: Set[str] = Field(default_factory=set)
    evidence_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    sections: Dict[str, str] = Field(default_factory=dict)
    unresolved_gaps: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 2
    final_report: Optional[str] = None
