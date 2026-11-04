"""MCP Tools package for DeepSearch-Local."""

from .models import SearchResult
from .scraper import fetch_page_markdown
from .search import SearchBackendError, web_search
from .ssrf import SSRFSecurityError, safe_fetch_with_redirect_check, validate_url_ssrf

__all__ = [
    "SearchResult",
    "SSRFSecurityError",
    "validate_url_ssrf",
    "safe_fetch_with_redirect_check",
    "fetch_page_markdown",
    "web_search",
    "SearchBackendError",
]
