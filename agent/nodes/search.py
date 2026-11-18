"""Search and Scrape node: Executes web queries and fetches Markdown with SSRF protection."""

from __future__ import annotations

import logging
from typing import Any

from agent.state import ResearchState
from mcp_server.tools.models import SearchResult
from mcp_server.tools.scraper import fetch_page_markdown
from mcp_server.tools.search import web_search

logger = logging.getLogger(__name__)


async def search_scrape_node(
    state: ResearchState,
    max_queries_per_step: int = 3,
    max_results_per_query: int = 4,
    max_pages_to_scrape: int = 5,
) -> dict[str, Any]:
    """Execute search queries and scrape newly discovered pages.

    Features:
    - Zero-overhead in-process tool execution.
    - URL deduplication against state.scraped_urls.
    - SSRF-guarded scraping via fetch_page_markdown.
    """
    logger.info(
        "Executing search_scrape_node (iteration=%d). Total queries in state=%d",
        state.iteration_count,
        len(state.search_queries),
    )

    # Determine which queries to execute: prioritize the most recent unexecuted queries
    queries_to_run = state.search_queries[-max_queries_per_step:] if state.search_queries else []
    if not queries_to_run:
        queries_to_run = [state.research_topic]

    discovered_results: list[SearchResult] = []
    seen_urls_in_batch: set[str] = set()

    # 1. Execute search queries concurrently with exception shielding
    for query in queries_to_run:
        try:
            results = await web_search(query=query, max_results=max_results_per_query)
            for res in results:
                if res.url not in state.scraped_urls and res.url not in seen_urls_in_batch:
                    discovered_results.append(res)
                    seen_urls_in_batch.add(res.url)
        except Exception as e:  # noqa: BLE001
            logger.warning("Search query failed for '%s': %s", query, e)

    # 2. Select top novel pages to scrape
    candidates_to_scrape = discovered_results[:max_pages_to_scrape]
    logger.info(
        "Discovered %d novel candidate URLs. Scraping top %d pages.",
        len(discovered_results),
        len(candidates_to_scrape),
    )

    raw_documents: list[dict[str, Any]] = []
    newly_scraped_urls: set[str] = set()

    for item in candidates_to_scrape:
        url = item.url
        title = item.title
        try:
            markdown = await fetch_page_markdown(url=url, max_length=50000, timeout=12.0)
            if markdown and not markdown.startswith("[Error:") and not markdown.startswith("[No readable"):
                raw_documents.append({
                    "url": url,
                    "title": title,
                    "content": markdown,
                })
                newly_scraped_urls.add(url)
            else:
                logger.debug("Skipping unreadable or errored page for URL %s: %s", url, markdown[:100] if markdown else "None")
                # Still record URL to avoid re-attempting failed URL
                newly_scraped_urls.add(url)
        except Exception as e:  # noqa: BLE001
            logger.warning("Scraping failed for URL %s: %s", url, e)
            newly_scraped_urls.add(url)

    logger.info(
        "search_scrape_node completed. Successfully extracted %d readable documents.",
        len(raw_documents),
    )

    return {
        "scraped_urls": newly_scraped_urls,
        "raw_documents": raw_documents,
    }
