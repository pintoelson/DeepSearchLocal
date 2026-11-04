"""Web scraper tool extracting clean Markdown using trafilatura and BeautifulSoup4 with SSRF protection."""

import re
from typing import Optional

from bs4 import BeautifulSoup
import httpx
import trafilatura

from .ssrf import SSRFSecurityError, safe_fetch_with_redirect_check, validate_url_ssrf

# Standard realistic browser headers to prevent basic anti-bot 403 Forbidden rejections
DEFAULT_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}

# Maximum byte size allowed for page download (5 MB) to prevent memory exhaustion / zip bombs
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024

# Allowed response content-type prefixes
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
    "text/xml",
    "application/xml",
)


def _extract_with_beautifulsoup(html_content: str) -> str:
    """Fallback extractor using BeautifulSoup when trafilatura yields no extractable content."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements and boilerplate containers
    for element in soup.find_all(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "noscript",
            "svg",
            "button",
            "iframe",
        ]
    ):
        element.decompose()

    # Extract text with newlines
    raw_text = soup.get_text(separator="\n", strip=True)

    # Normalize excessive consecutive blank lines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    return cleaned_text.strip()


async def fetch_page_markdown(
    url: str,
    max_length: int = 100_000,
    timeout: float = 15.0,
    headers: Optional[dict] = None,
) -> str:
    """Fetch a target webpage and extract clean Markdown content with strict SSRF protection.

    Args:
        url: The web URL to scrape.
        max_length: Maximum character length of returned Markdown text (default: 100,000).
        timeout: HTTP request timeout in seconds (default: 15.0).
        headers: Optional custom HTTP headers (defaults to realistic desktop browser headers).

    Returns:
        Clean, structured Markdown text extracted from the webpage.

    Raises:
        SSRFSecurityError: If the URL or any redirect hop targets a forbidden network or port.
        httpx.HTTPError: If the network request fails or returns an unrecoverable HTTP status.
    """
    # 1. Pre-validate URL before creating connection
    validate_url_ssrf(url)

    request_headers = {**DEFAULT_BROWSER_HEADERS, **(headers or {})}

    # 2. Async HTTP fetch with manual hop-by-hop redirect verification
    async with httpx.AsyncClient(
        headers=request_headers,
        timeout=httpx.Timeout(timeout, connect=10.0),
        verify=True,
    ) as client:
        response = await safe_fetch_with_redirect_check(
            client=client,
            url=url,
            max_redirects=5,
        )

        response.raise_for_status()

        # 3. Content-Type inspection
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(ct in content_type for ct in ALLOWED_CONTENT_TYPES):
            return f"[Error: Unsupported content type '{content_type}' for URL: {url}]"

        # 4. Content length guard
        if len(response.content) > MAX_DOWNLOAD_BYTES:
            return (
                f"[Error: Page size ({len(response.content)} bytes) exceeds max limit of "
                f"{MAX_DOWNLOAD_BYTES} bytes for URL: {url}]"
            )

        html_text = response.text

    # 5. Primary extraction: Trafilatura (State-of-the-art markdown extraction)
    markdown_content = trafilatura.extract(
        html_text,
        output_format="markdown",
        include_links=True,
        include_images=False,
        include_tables=True,
        no_fallback=False,
    )

    # 6. Fallback extraction: BeautifulSoup4
    if not markdown_content or not markdown_content.strip():
        markdown_content = _extract_with_beautifulsoup(html_text)

    if not markdown_content or not markdown_content.strip():
        return f"[No readable content could be extracted from: {url}]"

    # 7. Truncate to maximum requested length if needed
    if len(markdown_content) > max_length:
        truncated_marker = f"\n\n... [Content truncated at {max_length} characters]"
        return markdown_content[: max_length - len(truncated_marker)] + truncated_marker

    return markdown_content
