"""SSRF (Server-Side Request Forgery) protection and secure HTTP fetching utilities."""

import ipaddress
import socket
from typing import Any, List, Set, Union
from urllib.parse import urljoin, urlparse

import httpx


class SSRFSecurityError(Exception):
    """Raised when a requested URL violates SSRF safety boundaries."""

    pass


# Comprehensive list of private, link-local, loopback, and reserved network CIDRs
BLOCKED_NETWORKS: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = [
    # IPv4 loopback, private, and reserved ranges
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("10.0.0.0/8"),  # RFC 1918 Private Class A
    ipaddress.ip_network("172.16.0.0/12"),  # RFC 1918 Private Class B
    ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918 Private Class C
    ipaddress.ip_network("169.254.0.0/16"),  # RFC 3927 Link-Local / Cloud Metadata
    ipaddress.ip_network("0.0.0.0/8"),  # Current network / This host
    ipaddress.ip_network("100.64.0.0/10"),  # RFC 6598 Carrier-Grade NAT
    ipaddress.ip_network("192.0.0.0/24"),  # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),  # TEST-NET-1 documentation
    ipaddress.ip_network("198.18.0.0/15"),  # RFC 2544 Benchmark testing
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2 documentation
    ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3 documentation
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved for future use
    # IPv6 loopback, private, and reserved ranges
    ipaddress.ip_network("::1/128"),  # IPv6 Loopback
    ipaddress.ip_network("::/128"),  # IPv6 Unspecified
    ipaddress.ip_network("fc00::/7"),  # IPv6 Unique Local Address (ULA)
    ipaddress.ip_network("fe80::/10"),  # IPv6 Link-Local Unicast
    ipaddress.ip_network("ff00::/8"),  # IPv6 Multicast
    ipaddress.ip_network("2001:db8::/32"),  # IPv6 Documentation
]

BLOCKED_HOSTNAMES: Set[str] = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
}


def _is_ip_blocked(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    """Check if an IP address belongs to any banned or private network range."""
    # Handle IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped

    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True

    return any(ip in network for network in BLOCKED_NETWORKS)


def validate_url_ssrf(url: str) -> str:
    """Validate a URL against SSRF vulnerabilities by enforcing scheme, hostname, and IP restrictions.

    Args:
        url: The web URL to inspect.

    Returns:
        The normalized URL string if safe.

    Raises:
        SSRFSecurityError: If the URL uses an unapproved protocol or resolves to a private IP.
    """
    if not url or not isinstance(url, str):
        raise SSRFSecurityError("URL cannot be empty.")

    url = url.strip()
    parsed = urlparse(url)

    # 1. Enforce strict HTTP/HTTPS scheme
    if parsed.scheme.lower() not in ("http", "https"):
        raise SSRFSecurityError(
            f"Invalid scheme '{parsed.scheme}'. Only HTTP and HTTPS protocols are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise SSRFSecurityError(f"URL '{url}' does not contain a valid hostname.")

    hostname_lower = hostname.lower()

    # 2. Block well-known loopback hostnames
    if hostname_lower in BLOCKED_HOSTNAMES or hostname_lower.endswith(".localhost"):
        raise SSRFSecurityError(f"Access to loopback host '{hostname}' is forbidden.")

    # 3. Check if hostname is directly an IP literal
    try:
        # Strip square brackets for IPv6 literals
        ip_str = hostname.strip("[]")
        direct_ip = ipaddress.ip_address(ip_str)
        if _is_ip_blocked(direct_ip):
            raise SSRFSecurityError(
                f"Direct access to private/reserved IP '{direct_ip}' is forbidden."
            )
        return url
    except ValueError:
        # Not a raw IP literal; proceed with DNS resolution
        pass

    # 4. Resolve domain name to all bound IP addresses
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise SSRFSecurityError(f"Failed to resolve hostname '{hostname}': {e}") from e

    if not addr_info:
        raise SSRFSecurityError(f"No IP address records found for hostname '{hostname}'.")

    # 5. Validate every resolved IP
    for record in addr_info:
        sockaddr = record[4]
        ip_str = sockaddr[0]
        try:
            resolved_ip = ipaddress.ip_address(ip_str)
        except (ValueError, ipaddress.AddressValueError) as e:
            raise SSRFSecurityError(f"Invalid IP address returned from DNS: {ip_str}") from e

        if _is_ip_blocked(resolved_ip):
            raise SSRFSecurityError(
                f"Hostname '{hostname}' resolved to forbidden private/reserved IP: {ip_str}."
            )

    return url


async def safe_fetch_with_redirect_check(
    client: httpx.AsyncClient,
    url: str,
    max_redirects: int = 5,
    **kwargs: Any,
) -> httpx.Response:
    """Perform an async HTTP GET request while strictly validating each redirect hop against SSRF.

    Args:
        client: The httpx AsyncClient instance.
        url: The starting URL.
        max_redirects: Maximum allowed redirect hops before aborting.
        **kwargs: Additional parameters passed to client.get.

    Returns:
        The final HTTP Response object.

    Raises:
        SSRFSecurityError: If any hop in the redirect chain targets a forbidden network.
    """
    current_url = url
    hops_remaining = max_redirects

    while True:
        # Validate current URL before sending request
        validate_url_ssrf(current_url)

        # Force follow_redirects=False so we intercept each redirect manually
        response = await client.get(current_url, follow_redirects=False, **kwargs)

        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                # Malformed redirect without location header
                return response

            # Resolve relative URLs
            next_url = urljoin(current_url, location)

            if hops_remaining <= 0:
                raise SSRFSecurityError(
                    f"Redirect limit exceeded ({max_redirects} hops) while fetching {url}."
                )

            hops_remaining -= 1
            current_url = next_url
        else:
            return response
