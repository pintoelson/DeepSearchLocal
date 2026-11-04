"""Resilient web search tool supporting DuckDuckGo with exponential backoff, SearXNG, and Tavily fallbacks."""

import asyncio
import logging
import os
import random
from typing import List, Optional
import warnings

# Suppress the upstream rename warning for duckduckgo_search -> ddgs
warnings.filterwarnings("ignore", category=RuntimeWarning, message="This package .* has been renamed to .*ddgs")

from duckduckgo_search import DDGS
import httpx

from .models import SearchResult

logger = logging.getLogger(__name__)


class SearchBackendError(RuntimeError):
    """Raised when all configured search providers fail to return results."""

    pass


def _sync_ddg_search(query: str, max_results: int) -> List[SearchResult]:
    """Synchronous worker function executing DuckDuckGo search with multi-backend fallback."""
    raw_results = []
    # Try primary auto backend; if rate-limited / empty, attempt the lightweight 'lite' backend
    for backend in ("auto", "lite"):
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results, backend=backend))
            if raw_results:
                break
        except Exception as e:
            logger.debug(f"DDGS backend '{backend}' encountered exception: {e}")
            continue

    results: List[SearchResult] = []
    for item in raw_results:
        url = item.get("href") or item.get("url") or ""
        title = item.get("title") or "Untitled"
        snippet = item.get("body") or item.get("snippet") or ""

        if url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results


async def _search_duckduckgo_with_backoff(
    query: str,
    max_results: int = 5,
    max_retries: int = 3,
    base_backoff_sec: float = 1.0,
) -> List[SearchResult]:
    """Execute DuckDuckGo search with exponential backoff and randomized jitter."""
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            results = await asyncio.to_thread(_sync_ddg_search, query, max_results)
            if results:
                return results
            # Empty results on first attempt could just mean no matches, return empty list
            return []
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff formula with jitter: base * 2^attempt + uniform(0, 0.5)
                sleep_duration = (base_backoff_sec * (2**attempt)) + random.uniform(0.1, 0.5)
                logger.warning(
                    f"DuckDuckGo search attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {sleep_duration:.2f}s..."
                )
                await asyncio.sleep(sleep_duration)
            else:
                logger.error(f"DuckDuckGo search exhausted {max_retries + 1} attempts for '{query}'.")

    raise SearchBackendError(f"DuckDuckGo failed: {last_exception}") from last_exception


async def _search_searxng(
    query: str,
    base_url: str,
    max_results: int = 5,
    timeout: float = 10.0,
) -> List[SearchResult]:
    """Execute search query against a self-hosted or remote SearXNG instance."""
    clean_base = base_url.rstrip("/")
    search_url = f"{clean_base}/search"

    params = {
        "q": query,
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()

    results: List[SearchResult] = []
    for item in data.get("results", [])[:max_results]:
        url = item.get("url", "")
        title = item.get("title", "Untitled")
        snippet = item.get("content", "") or item.get("snippet", "")
        if url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results


async def _search_tavily(
    query: str,
    api_key: str,
    max_results: int = 5,
    timeout: float = 10.0,
) -> List[SearchResult]:
    """Execute search query against Tavily Search API."""
    endpoint = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()

    results: List[SearchResult] = []
    for item in data.get("results", [])[:max_results]:
        url = item.get("url", "")
        title = item.get("title", "Untitled")
        snippet = item.get("content", "") or item.get("snippet", "")
        if url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results


async def web_search(
    query: str,
    max_results: int = 5,
    max_retries: int = 3,
) -> List[SearchResult]:
    """Execute a resilient web search with automatic fallback routing.

    Hierarchy:
        1. DuckDuckGo Search (primary, zero-cost, exponential backoff)
        2. SearXNG Fallback (if SEARXNG_BASE_URL is set in environment)
        3. Tavily Search Fallback (if TAVILY_API_KEY is set in environment)

    Args:
        query: Search string or keywords.
        max_results: Maximum number of search results to return (default: 5).
        max_retries: Retries for DuckDuckGo backoff (default: 3).

    Returns:
        List of SearchResult objects containing title, URL, and snippet.

    Raises:
        SearchBackendError: If all primary and configured fallback providers fail.
    """
    if not query or not query.strip():
        return []

    query = query.strip()
    errors = []

    # 1. Primary: DuckDuckGo with exponential backoff
    try:
        return await _search_duckduckgo_with_backoff(
            query=query,
            max_results=max_results,
            max_retries=max_retries,
        )
    except Exception as ddg_err:
        logger.warning(f"DuckDuckGo search failed for query '{query}': {ddg_err}. Checking fallbacks...")
        errors.append(f"DuckDuckGo: {ddg_err}")

    # 2. Fallback 1: SearXNG (if configured)
    searxng_url = os.environ.get("SEARXNG_BASE_URL")
    if searxng_url:
        try:
            logger.info(f"Attempting SearXNG fallback at {searxng_url} for '{query}'...")
            return await _search_searxng(query=query, base_url=searxng_url, max_results=max_results)
        except Exception as searx_err:
            logger.warning(f"SearXNG fallback failed: {searx_err}")
            errors.append(f"SearXNG: {searx_err}")

    # 3. Fallback 2: Tavily (if configured)
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            logger.info(f"Attempting Tavily fallback for '{query}'...")
            return await _search_tavily(query=query, api_key=tavily_key, max_results=max_results)
        except Exception as tavily_err:
            logger.warning(f"Tavily fallback failed: {tavily_err}")
            errors.append(f"Tavily: {tavily_err}")

    # All attempted backends failed
    error_summary = "; ".join(errors)
    raise SearchBackendError(
        f"All search backends failed for query '{query}'. Errors: {error_summary}"
    )
