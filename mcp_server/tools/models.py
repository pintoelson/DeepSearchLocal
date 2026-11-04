"""Data models for DeepSearch-Local Tool Plane."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Structured representation of a single web search result."""

    title: str = Field(..., description="Headline or title of the webpage.")
    url: str = Field(..., description="Direct HTTP/HTTPS URL of the webpage.")
    snippet: str = Field(
        default="",
        description="Text excerpt or summary snippet returned by the search engine.",
    )
