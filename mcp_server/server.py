"""Model Context Protocol (MCP) JSON-RPC Server interface."""

import argparse
import os
import sys
from typing import Any, Dict, List

from mcp.server.mcpserver import MCPServer
from mcp_server.tools import fetch_page_markdown, web_search


def create_mcp_server() -> MCPServer:
    """Instantiate and configure the MCP Server exposing search and scraping tools."""
    server = MCPServer(
        name="deepsearch-tools",
        version="0.1.0",
        description="Autonomous Deep Research Agent tool plane (search, SSRF-guarded scraping)",
    )

    @server.tool(
        name="web_search",
        description=(
            "Execute a web search using DuckDuckGo (with automatic SearXNG/Tavily fallback) "
            "and return structured results containing title, URL, and snippet."
        ),
    )
    async def handle_web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for research evidence and external facts.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve (default: 5).
        """
        results = await web_search(query=query, max_results=max_results)
        return [r.model_dump() for r in results]

    @server.tool(
        name="fetch_page_markdown",
        description=(
            "Fetch a webpage and extract clean, readable Markdown using Trafilatura "
            "with BeautifulSoup fallback and strict SSRF protection."
        ),
    )
    async def handle_fetch_page_markdown(url: str, max_length: int = 100000) -> str:
        """Fetch page content converted to Markdown.

        Args:
            url: Target web page URL.
            max_length: Maximum allowed characters in returned markdown.
        """
        return await fetch_page_markdown(url=url, max_length=max_length)

    return server


def main() -> None:
    """Entry point for running the standalone MCP server via CLI."""
    parser = argparse.ArgumentParser(
        description="DeepSearch-Local Model Context Protocol (MCP) Server"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="Transport layer protocol: 'stdio' (default, for Claude Desktop/Cursor) or 'sse'",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_SERVER_HOST", "127.0.0.1"),
        help="Host address to bind for SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_SERVER_PORT", "8001")),
        help="Port to bind for SSE transport (default: 8001)",
    )

    args = parser.parse_args()
    server = create_mcp_server()

    if args.transport == "stdio":
        server.run(transport="stdio")
    elif args.transport == "sse":
        server.run(transport="sse", host=args.host, port=args.port)
    else:
        sys.stderr.write(f"Unsupported transport: {args.transport}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
