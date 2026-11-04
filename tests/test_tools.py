"""Comprehensive unit and integration tests for Section 3.2 Tool Plane."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mcp_server.server import create_mcp_server
from mcp_server.tools.models import SearchResult
from mcp_server.tools.scraper import fetch_page_markdown
from mcp_server.tools.search import SearchBackendError, web_search
from mcp_server.tools.ssrf import (
    SSRFSecurityError,
    safe_fetch_with_redirect_check,
    validate_url_ssrf,
)


# ==============================================================================
# SSRF Protection Tests
# ==============================================================================


def test_ssrf_rejects_empty_and_non_http_schemes():
    """Verify SSRF validator rejects empty strings and non-http schemes."""
    with pytest.raises(SSRFSecurityError, match="URL cannot be empty"):
        validate_url_ssrf("")

    with pytest.raises(SSRFSecurityError, match="Only HTTP and HTTPS"):
        validate_url_ssrf("file:///etc/passwd")

    with pytest.raises(SSRFSecurityError, match="Only HTTP and HTTPS"):
        validate_url_ssrf("gopher://127.0.0.1:70")

    with pytest.raises(SSRFSecurityError, match="Only HTTP and HTTPS"):
        validate_url_ssrf("ftp://example.com/file.txt")


def test_ssrf_rejects_loopback_and_local_hostnames():
    """Verify SSRF validator blocks localhost, 127.0.0.1, and .localhost domains."""
    blocked_hosts = [
        "http://localhost:8000",
        "http://localhost.localdomain",
        "http://test.localhost",
        "http://127.0.0.1:6333",
        "http://127.0.1.1:11434",
        "http://127.255.255.254",
    ]
    for url in blocked_hosts:
        with pytest.raises(SSRFSecurityError):
            validate_url_ssrf(url)


def test_ssrf_rejects_private_ipv4_and_cloud_metadata():
    """Verify SSRF validator blocks RFC 1918 private IPs and cloud metadata IP (169.254.169.254)."""
    private_urls = [
        "http://10.0.0.1/admin",
        "http://10.255.255.255",
        "http://172.16.0.1:8080",
        "http://172.31.255.254",
        "http://192.168.1.1/router",
        "http://192.168.0.254",
        "http://169.254.169.254/latest/meta-data",  # AWS / GCP / Azure metadata endpoint
        "http://0.0.0.0:8000",
        "http://100.64.0.1",  # Carrier-grade NAT
    ]
    for url in private_urls:
        with pytest.raises(SSRFSecurityError):
            validate_url_ssrf(url)


def test_ssrf_rejects_ipv6_loopback_and_link_local():
    """Verify SSRF validator blocks IPv6 loopback, ULA, and link-local literals."""
    ipv6_urls = [
        "http://[::1]:8080",
        "http://[::]:8000",
        "http://[fe80::1]:80",
        "http://[fc00::1]:8080",
    ]
    for url in ipv6_urls:
        with pytest.raises(SSRFSecurityError):
            validate_url_ssrf(url)


def test_ssrf_allows_public_ip_literal():
    """Verify SSRF validator allows public, non-banned IP addresses."""
    # Google Public DNS IPv4
    assert validate_url_ssrf("http://8.8.8.8") == "http://8.8.8.8"
    assert validate_url_ssrf("https://1.1.1.1/dns-query") == "https://1.1.1.1/dns-query"


def test_ssrf_rejects_domain_resolving_to_private_ip():
    """Verify SSRF validator blocks domain names that resolve via DNS to private IPs."""
    # Simulate a domain that resolves to 127.0.0.1 (e.g. DNS rebinding)
    fake_addr_info = [
        (2, 1, 6, "", ("127.0.0.1", 80)),
    ]
    with patch("socket.getaddrinfo", return_value=fake_addr_info):
        with pytest.raises(SSRFSecurityError, match="forbidden private/reserved IP"):
            validate_url_ssrf("http://rebind.attacker.com")


@pytest.mark.asyncio
async def test_ssrf_redirect_hop_protection():
    """Verify that a redirect from a public URL to an internal private IP is intercepted and blocked."""
    # First response: 302 redirecting to http://127.0.0.1:6333/collections
    redirect_resp = httpx.Response(
        status_code=302,
        headers={"Location": "http://127.0.0.1:6333/collections"},
        request=httpx.Request("GET", "https://example.com/safe"),
    )

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = redirect_resp

    with patch("mcp_server.tools.ssrf.validate_url_ssrf") as mock_validate:
        # First call for example.com passes; second call for 127.0.0.1 raises SSRFSecurityError
        mock_validate.side_effect = [
            "https://example.com/safe",
            SSRFSecurityError("Blocked private IP 127.0.0.1"),
        ]

        with pytest.raises(SSRFSecurityError, match="Blocked private IP"):
            await safe_fetch_with_redirect_check(client, "https://example.com/safe")


@pytest.mark.asyncio
async def test_ssrf_redirect_limit_exceeded():
    """Verify redirect loops or excessive hops trigger an SSRFSecurityError."""
    loop_resp = httpx.Response(
        status_code=302,
        headers={"Location": "https://example.com/loop"},
        request=httpx.Request("GET", "https://example.com/loop"),
    )

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = loop_resp

    with patch("mcp_server.tools.ssrf.validate_url_ssrf", return_value="https://example.com/loop"):
        with pytest.raises(SSRFSecurityError, match="Redirect limit exceeded"):
            await safe_fetch_with_redirect_check(
                client, "https://example.com/loop", max_redirects=3
            )


# ==============================================================================
# Web Scraper Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_fetch_page_markdown_trafilatura_extraction():
    """Verify clean markdown extraction from normal article HTML."""
    html_sample = """
    <!DOCTYPE html>
    <html>
      <head><title>Research Breakthrough</title></head>
      <body>
        <nav><a href="/home">Home</a><a href="/about">About</a></nav>
        <article>
          <h1>Breakthrough in Quantum Computing</h1>
          <p>Researchers achieved 99.9% gate fidelity using superconducting transmon qubits.</p>
        </article>
        <footer><p>Copyright 2026</p></footer>
      </body>
    </html>
    """

    mock_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=html_sample,
        request=httpx.Request("GET", "https://research.example.com/article"),
    )

    with (
        patch("mcp_server.tools.scraper.validate_url_ssrf"),
        patch(
            "mcp_server.tools.scraper.safe_fetch_with_redirect_check",
            return_value=mock_resp,
        ),
    ):
        result = await fetch_page_markdown("https://research.example.com/article")
        assert "Breakthrough in Quantum Computing" in result
        assert "superconducting transmon qubits" in result
        # Boilerplate nav should be removed
        assert "Home" not in result


@pytest.mark.asyncio
async def test_fetch_page_markdown_beautifulsoup_fallback():
    """Verify fallback to BeautifulSoup when trafilatura returns None/empty."""
    html_sample = """
    <html>
      <body>
        <script>console.log("ad code");</script>
        <style>.hide { display: none; }</style>
        <div class="custom-body">
          <p>Important fallback documentation text.</p>
        </div>
      </body>
    </html>
    """

    mock_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "text/html"},
        text=html_sample,
        request=httpx.Request("GET", "https://example.com/tricky"),
    )

    with (
        patch("mcp_server.tools.scraper.validate_url_ssrf"),
        patch(
            "mcp_server.tools.scraper.safe_fetch_with_redirect_check",
            return_value=mock_resp,
        ),
        patch("trafilatura.extract", return_value=None),  # Simulate trafilatura failing
    ):
        result = await fetch_page_markdown("https://example.com/tricky")
        assert "Important fallback documentation text." in result
        assert "console.log" not in result  # script tag decomposed


@pytest.mark.asyncio
async def test_fetch_page_markdown_unsupported_content_type():
    """Verify rejection of binary content types like PDF or PNG."""
    mock_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "application/pdf"},
        content=b"%PDF-1.4...",
        request=httpx.Request("GET", "https://example.com/document.pdf"),
    )

    with (
        patch("mcp_server.tools.scraper.validate_url_ssrf"),
        patch(
            "mcp_server.tools.scraper.safe_fetch_with_redirect_check",
            return_value=mock_resp,
        ),
    ):
        result = await fetch_page_markdown("https://example.com/document.pdf")
        assert "Unsupported content type" in result


@pytest.mark.asyncio
async def test_fetch_page_markdown_truncates_long_content():
    """Verify truncation marker is appended when content exceeds max_length."""
    long_html = f"<html><body><p>{'A' * 2000}</p></body></html>"
    mock_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "text/html"},
        text=long_html,
        request=httpx.Request("GET", "https://example.com/long"),
    )

    with (
        patch("mcp_server.tools.scraper.validate_url_ssrf"),
        patch(
            "mcp_server.tools.scraper.safe_fetch_with_redirect_check",
            return_value=mock_resp,
        ),
    ):
        result = await fetch_page_markdown("https://example.com/long", max_length=500)
        assert len(result) <= 500
        assert "Content truncated" in result


# ==============================================================================
# Web Search Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_web_search_empty_query():
    """Verify empty search query returns empty list immediately."""
    assert await web_search("") == []
    assert await web_search("   ") == []


@pytest.mark.asyncio
async def test_web_search_duckduckgo_success():
    """Verify DuckDuckGo search maps raw items to SearchResult objects."""
    mock_raw_items = [
        {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
        {"title": "Result 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
    ]

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = mock_raw_items
    mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.__exit__.return_value = None

    with patch("mcp_server.tools.search.DDGS", return_value=mock_ddgs_instance):
        results = await web_search("python deep research", max_results=2)
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "Result 1"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "Snippet 1"
        assert results[1].title == "Result 2"
        assert results[1].url == "https://example.com/2"
        assert results[1].snippet == "Snippet 2"


@pytest.mark.asyncio
async def test_web_search_duckduckgo_exponential_backoff_retry():
    """Verify DuckDuckGo retries with backoff upon encountering exceptions."""
    mock_calls = 0

    def faulty_ddg(query, max_results):
        nonlocal mock_calls
        mock_calls += 1
        if mock_calls < 3:
            raise RuntimeError("Rate limited 202")
        return [SearchResult(title="Finally Recovered", url="https://example.com", snippet="OK")]

    with (
        patch("mcp_server.tools.search._sync_ddg_search", side_effect=faulty_ddg),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        results = await web_search("test query", max_retries=3)
        assert len(results) == 1
        assert results[0].title == "Finally Recovered"
        assert mock_calls == 3
        # Should have slept twice (after attempt 1 and attempt 2)
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_web_search_searxng_fallback():
    """Verify fallback to SearXNG when DuckDuckGo fails and SEARXNG_BASE_URL is configured."""
    with (
        patch("mcp_server.tools.search._sync_ddg_search", side_effect=RuntimeError("DDG Blocked")),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch.dict(os.environ, {"SEARXNG_BASE_URL": "http://searxng.local:8080"}),
        patch(
            "mcp_server.tools.search._search_searxng",
            new_callable=AsyncMock,
            return_value=[
                SearchResult(
                    title="SearXNG Match",
                    url="https://searx.example.com",
                    snippet="Found via SearXNG",
                )
            ],
        ),
    ):
        results = await web_search("autonomous agents", max_results=1)
        assert len(results) == 1
        assert results[0].title == "SearXNG Match"


@pytest.mark.asyncio
async def test_web_search_tavily_fallback():
    """Verify fallback to Tavily when DuckDuckGo fails and TAVILY_API_KEY is configured."""
    with (
        patch("mcp_server.tools.search._sync_ddg_search", side_effect=RuntimeError("DDG Blocked")),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-secret-key"}, clear=False),
        patch(
            "mcp_server.tools.search._search_tavily",
            new_callable=AsyncMock,
            return_value=[
                SearchResult(
                    title="Tavily Match",
                    url="https://tavily.example.com",
                    snippet="Found via Tavily",
                )
            ],
        ),
    ):
        # Ensure SEARXNG is unset
        os.environ.pop("SEARXNG_BASE_URL", None)
        results = await web_search("langgraph state machines", max_results=1)
        assert len(results) == 1
        assert results[0].title == "Tavily Match"


@pytest.mark.asyncio
async def test_web_search_all_fail_raises_search_backend_error():
    """Verify SearchBackendError is raised when all search options fail."""
    with (
        patch("mcp_server.tools.search._sync_ddg_search", side_effect=RuntimeError("DDG Down")),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch.dict(os.environ, {}, clear=True),
    ):
        with pytest.raises(SearchBackendError, match="All search backends failed"):
            await web_search("critical search query", max_retries=1)


# ==============================================================================
# Standalone MCP Server Integration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_mcp_server_tools_registered():
    """Verify create_mcp_server registers web_search and fetch_page_markdown tools."""
    server = create_mcp_server()
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    assert "web_search" in tool_names
    assert "fetch_page_markdown" in tool_names


@pytest.mark.asyncio
async def test_mcp_server_call_web_search():
    """Verify calling web_search via the MCP server interface executes correctly."""
    server = create_mcp_server()
    mock_results = [SearchResult(title="MCP Doc", url="https://mcp.io", snippet="MCP Specification")]

    with patch("mcp_server.server.web_search", new_callable=AsyncMock, return_value=mock_results):
        response = await server.call_tool("web_search", {"query": "model context protocol"})
        assert not response.is_error
        # Verify JSON structured output
        assert len(response.structured_content["result"]) == 1
        assert response.structured_content["result"][0]["title"] == "MCP Doc"


@pytest.mark.asyncio
async def test_mcp_server_call_fetch_page_markdown():
    """Verify calling fetch_page_markdown via the MCP server interface executes correctly."""
    server = create_mcp_server()

    with patch(
        "mcp_server.server.fetch_page_markdown",
        new_callable=AsyncMock,
        return_value="# Page Title\n\nPage Body.",
    ):
        response = await server.call_tool(
            "fetch_page_markdown", {"url": "https://mcp.io/spec"}
        )
        assert not response.is_error
        assert "# Page Title" in response.structured_content["result"]
