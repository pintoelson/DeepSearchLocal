# DeepSearch-Local: Tool Layer, SSRF Security & Model Context Protocol (MCP) Guide

This guide is an end-to-end educational reference designed for engineers learning **Agent Tool Execution**, **SSRF (Server-Side Request Forgery) Defense-in-Depth**, **Resilient Web Retrieval**, and the **Anthropic Model Context Protocol (MCP)**.

It explains the exact networking constraints, security vulnerabilities, latency trade-offs, and tokenization economics implemented in `DeepSearch-Local`'s tool layer.

---

## Table of Contents
1. [The High-Level Objective](#1-the-high-level-objective)
2. [Dual-Execution Architecture: In-Process vs. Standalone MCP](#2-dual-execution-architecture-in-process-vs-standalone-mcp)
3. [Search Resilience: Rate Limits, Exponential Backoff & Fallbacks](#3-search-resilience-rate-limits-exponential-backoff--fallbacks)
4. [Hardened SSRF Defense-in-Depth](#4-hardened-ssrf-defense-in-depth)
5. [Content Extraction Pipeline & Token Economics](#5-content-extraction-pipeline--token-economics)
6. [Anthropic Model Context Protocol (MCP 2.x) Integration](#6-anthropic-model-context-protocol-mcp-2x-integration)
7. [Step-by-Step Hands-On Guide](#7-step-by-step-hands-on-guide)

---

## 1. The High-Level Objective

In an autonomous deep research platform, Small Language Models (SLMs, 1.5B–3B parameters) cannot rely solely on parametric knowledge. If asked to summarize recent breakthroughs or synthesize factual reports without external context, SLMs frequently hallucinate or produce outdated claims.

The **Tool Layer** acts as the agent's sensory organs, providing two foundational capabilities:
1. **Targeted Web Discovery (`web_search`)**: Identifying authoritative URLs and passages matching decomposed search queries.
2. **Sanitized Content Extraction (`fetch_page_markdown`)**: Ingesting web pages, stripping noise and boilerplate, and formatting clean Markdown text for the RAG embedding pipeline.

```
┌────────────────────────────────────────────────────────┐
│             LangGraph Cyclic State Machine             │
│                                                        │
│   Planner Node ──► Search & Scrape ──► Ingestion Node  │
│        ▲                   │                 │         │
│        │                   ▼                 ▼         │
│        └──────────── Synthesis Loop ◄── Qdrant (CPU)   │
└────────────────────────────┼───────────────────────────┘
                             │ Direct Python calls
                             ▼
┌────────────────────────────────────────────────────────┐
│               DeepSearch-Local Tool Plane              │
│                                                        │
│  ┌───────────────────────┐   ┌──────────────────────┐  │
│  │      web_search       │   │ fetch_page_markdown  │  │
│  │ (DDG + Backoff/Jitter │   │ (Trafilatura + BS4 + │  │
│  │ + SearXNG/Tavily)     │   │  Hardened SSRF Guard)│  │
│  └───────────────────────┘   └──────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 2. Dual-Execution Architecture: In-Process vs. Standalone MCP

A common design pitfall in modern AI agent development is forcing internal agent loops to communicate with their own tools over network sockets or inter-process pipes (IPC).

### Latency Arithmetic: In-Process vs. IPC vs. HTTP
| Execution Mode | Transport | Serialization Overhead | Typical Latency | Failure Modes |
| :--- | :--- | :--- | :--- | :--- |
| **Native In-Process** | Python Function Call | **$0\text{ ms}$** (direct memory reference) | **$< 0.001\text{ ms}$** | Direct Python exceptions |
| **Local Subprocess / Stdio** | Pipe / JSON-RPC | $0.5 - 2\text{ ms}$ (JSON encode/decode) | $2 - 10\text{ ms}$ | Broken pipes, buffer deadlocks |
| **Local HTTP / SSE** | TCP Loopback Socket | $1 - 3\text{ ms}$ (HTTP parsing + JSON) | $5 - 20\text{ ms}$ | Port conflicts, connection resets |

In a deep research session executing 10 search queries and scraping 25 web pages across 3 reflection cycles:
- **In-Process**: Zero transport overhead, zero socket descriptor consumption, and full native Python exception stack traces.
- **Why build an MCP Server then?** Interoperability. Developers working in tools like **Claude Desktop**, **Cursor**, or **Zed** can reuse the exact same validated search and scraping tools without launching the full DeepSearch web dashboard.

### The Unified Architecture
To achieve both zero overhead and universal compatibility, we write the tools once as clean asynchronous Python functions in `mcp_server/tools/`:
- `agent/nodes/search_scrape_node.py` imports and invokes the functions directly.
- `mcp_server/server.py` wraps them in the Anthropic `mcp` SDK to expose a JSON-RPC interface.

---

## 3. Search Resilience: Rate Limits, Exponential Backoff & Fallbacks

### The Rate Limit Challenge
DuckDuckGo (`duckduckgo-search`) is an ideal zero-cost search provider because it requires no API keys and charges no recurring fees. However, DDG aggressively throttles IP addresses that issue rapid sequential search queries, returning HTTP 202 or rate-limit exceptions.

### Exponential Backoff with Jitter
To prevent transient throttles from crashing the agent, the search engine retries failed requests using **truncated exponential backoff with full randomized jitter**:

$$T_{\text{wait}} = B \times 2^{\text{attempt}} + \mathcal{U}(0, J)$$

where:
- $B = 1.0\text{ s}$ (base backoff duration)
- $\text{attempt} \in \{0, 1, 2\}$
- $\mathcal{U}(0, J)$ is a uniform random variable with $J = 0.5\text{ s}$ (decorrelating simultaneous worker threads to avoid the "thundering herd" problem)

```
Attempt 1 (Immediate) ──► Fail (429/202)
   │
   ▼ Wait 1.0s + jitter (~1.2s)
Attempt 2 ───────────────► Fail (429/202)
   │
   ▼ Wait 2.0s + jitter (~2.3s)
Attempt 3 ───────────────► Fail (429/202)
   │
   ▼ Exhausted: Transition to Fallback Engine
```

### Dual Fallback Routing
If DuckDuckGo fails all retry attempts:
1. **Fallback 1: SearXNG**: If `SEARXNG_BASE_URL` is configured, routes the query to a local or self-hosted SearXNG meta-search engine.
2. **Fallback 2: Tavily**: If `TAVILY_API_KEY` is configured, routes the query to the Tavily API.
3. If neither fallback is configured or all fail, a structured `SearchBackendError` is raised.

---

## 4. Hardened SSRF Defense-in-Depth

### The Threat Model for Local AI Systems
Local AI environments are uniquely vulnerable to **Server-Side Request Forgery (SSRF)**. Unlike cloud microservices isolated in VPCs, your local machine typically runs other sensitive services on `localhost`:
- **Qdrant Vector DB**: Listening on `http://127.0.0.1:6333` (unauthenticated REST API allowing arbitrary collection deletion and data exfiltration).
- **Ollama Engine**: Listening on `http://127.0.0.1:11434` (unauthenticated local LLM control and model deletion).
- **Local Network / Router**: Gateway management interfaces at `192.168.1.1` or `10.0.0.1`.
- **Cloud Metadata (if deployed in cloud)**: AWS/GCP instance metadata at `169.254.169.254`.

If an autonomous agent scrapes a URL suggested by an untrusted source or search engine, an attacker could supply `http://127.0.0.1:6333/collections` or use a **302 Redirect Attack**:

```
[Agent Scraper] ───────────────► GET http://attacker.com/article
                                      │
                                      ▼ Attacker returns HTTP 302
                                 Location: http://127.0.0.1:6333/collections/delete
                                      │
[Naive Scraper (follow_redirects=True)] ──► BLINDLY DELETES LOCAL QDRANT!
```

### Our Multi-Hop Defense Pipeline
Our scraper protects against direct attacks, DNS rebinding, and redirect chaining:

```
Target URL ──► 1. Protocol Scheme Check (http/https only)
                     │
                     ▼
               2. Hostname Check (reject localhost, *.localhost)
                     │
                     ▼
               3. DNS Pre-Resolution (socket.getaddrinfo)
                     │
                     ▼
               4. CIDR Boundary Check (RFC 1918, RFC 3927, IPv6 ULA)
                     │ (Safe IP)
                     ▼
               5. HTTP Fetch with follow_redirects=False
                     │
             Is 3xx Redirect?
             ├── No  ──► Return Safe Response Body
             └── Yes ──► Extract Location Header
                             │
                             ▼
                         Loop back to Step 1 (Max 5 hops)
```

### Blocked CIDR Specifications
Every resolved IP is evaluated against:
- `127.0.0.0/8` (Loopback)
- `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` (Private RFC 1918)
- `169.254.0.0/16` (Link-local RFC 3927 & Cloud Metadata)
- `0.0.0.0/8` (Current network / "this host")
- `100.64.0.0/10` (Carrier-grade NAT RFC 6598)
- `::1/128` (IPv6 Loopback)
- `fc00::/7` (IPv6 Unique Local Address RFC 4193)
- `fe80::/10` (IPv6 Link-Local Unicast RFC 4291)

---

## 5. Content Extraction Pipeline & Token Economics

### The Token Problem
A typical modern web page contains **50,000 to 150,000 tokens** of raw HTML:
- JavaScript bundles (`<script>`)
- CSS stylesheets (`<style>`)
- Navigation menus, cookie consent banners, footers, advertisement trackers

Passing raw HTML into an SLM with a **4,096-token context window** causes instant context overflow and wastes 95% of GPU memory on useless markup.

### Trafilatura: Algorithmic Boilerplate Stripping
`trafilatura` computes the ratio of text density to DOM tree complexity, isolating the main editorial content. It produces clean, semantic Markdown:
- Headings (`#`, `##`) are preserved for chunk header metadata.
- Links are retained (`[Anchor](url)`) for source grounding.
- Images, ads, and interactive scripts are stripped.

$$\text{Compression Ratio} = 1 - \frac{\text{Markdown Tokens}}{\text{Raw HTML Tokens}} \approx 92\% - 97\%$$

### BeautifulSoup4 Fallback
Certain dynamic web pages or non-standard HTML layouts cause Trafilatura's heuristic parser to return an empty string. Rather than discarding the page, our pipeline immediately falls back to `BeautifulSoup4`:
1. Decomposes non-content tags: `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>`, `<form>`, `<noscript>`, `<svg>`.
2. Extracts structured text with whitespace normalization.
3. Enforces a maximum character cap (`100,000` chars) and download size cap (`5 MB`) to prevent memory exhaustion and zip-bomb attacks.

---

## 6. Anthropic Model Context Protocol (MCP 2.x) Integration

DeepSearch-Local provides a standalone Model Context Protocol server using the official Anthropic `mcp` SDK (`mcp.server.mcpserver.MCPServer`).

### Exposed Tools Schema
1. **`web_search`**:
   - Parameters: `query` (string, required), `max_results` (integer, default: 5).
   - Returns: List of objects containing `title`, `url`, `snippet`.
2. **`fetch_page_markdown`**:
   - Parameters: `url` (string, required), `max_length` (integer, default: 100000).
   - Returns: Markdown text string.

### Connecting External Clients (Claude Desktop / Cursor)
To expose DeepSearch's tools to **Claude Desktop**, add the following configuration to `~/.config/Claude/claude_desktop_config.json` (Linux) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "deepsearch-tools": {
      "command": "/home/elson/Files/DeepSearchLocal/.venv/bin/python",
      "args": [
        "-m",
        "mcp_server.server",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

---

## 7. Step-by-Step Hands-On Guide

### Step 1: Run the Automated Tool Test Suite
Verify that all 21 tool tests (SSRF guards, DuckDuckGo search, backoff retries, BeautifulSoup fallbacks, and MCP schemas) pass:
```bash
.venv/bin/pytest tests/test_tools.py -v
```

### Step 2: Test Standalone Search in Python
Execute an in-process search query:
```bash
.venv/bin/python -c "
import asyncio
from mcp_server.tools import web_search

async def run():
    results = await web_search('Qwen 2.5 3B instruct benchmark', max_results=3)
    for i, r in enumerate(results, 1):
        print(f'{i}. {r.title} -> {r.url}')

asyncio.run(run())
"
```

### Step 3: Test SSRF Protection
Verify that the scraper strictly blocks requests targeting internal local infrastructure:
```bash
.venv/bin/python -c "
import asyncio
from mcp_server.tools import fetch_page_markdown, SSRFSecurityError

async def test():
    try:
        await fetch_page_markdown('http://127.0.0.1:6333')
        print('FAILED: Allowed private IP!')
    except SSRFSecurityError as e:
        print(f'SUCCESS: Blocked private IP -> {e}')

asyncio.run(test())
"
```

### Step 4: Run the Standalone MCP Server
Start the MCP server over stdio for local client inspection:
```bash
.venv/bin/python -m mcp_server.server --help
```

To run as an SSE (Server-Sent Events) microservice on port 8001:
```bash
.venv/bin/python -m mcp_server.server --transport sse --host 127.0.0.1 --port 8001
```
