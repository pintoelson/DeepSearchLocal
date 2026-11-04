"""Deterministic syntactic and citation verification metrics for DeepSearch evaluation."""

from __future__ import annotations

import json
import re
from typing import Any


# ==============================================================================
# 1. Tool Calling Metrics
# ==============================================================================


def extract_json_payload(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract and parse JSON from raw model generation (plain or code-fenced)."""
    text = text.strip()
    if not text:
        return None

    # 1. Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try markdown fenced codeblock ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        code_content = match.group(1).strip()
        try:
            return json.loads(code_content)
        except json.JSONDecodeError:
            pass

    # 3. Try finding outermost { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = text[first_brace : last_brace + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    return None


def evaluate_tool_calling(
    response_text: str,
    allowed_tools: list[str] | None = None,
    required_tool: str | None = None,
) -> dict[str, Any]:
    """Evaluate JSON validity and tool schema conformity.

    Returns dict with:
        - is_valid_json: bool
        - has_valid_schema: bool
        - tool_name: str | None
        - is_allowed_tool: bool
        - matches_required_tool: bool
        - score: float (0.0 to 1.0)
    """
    allowed_tools = allowed_tools or ["web_search", "fetch_page_markdown"]
    payload = extract_json_payload(response_text)

    if payload is None or not isinstance(payload, dict):
        return {
            "is_valid_json": False,
            "has_valid_schema": False,
            "tool_name": None,
            "is_allowed_tool": False,
            "matches_required_tool": False,
            "score": 0.0,
            "error": "Failed to parse valid JSON dictionary",
        }

    # Extract tool name across common schema conventions
    # DeepSearch format: {"tool": "web_search", "arguments": {"query": "..."}}
    # OpenAI format: {"name": "web_search", "arguments": {...}}
    tool_name = payload.get("tool") or payload.get("name") or payload.get("function")
    arguments = payload.get("arguments") or payload.get("parameters")

    # If arguments are at the top level (e.g. {"web_search": {"query": "..."}})
    if not tool_name:
        for t in allowed_tools:
            if t in payload and isinstance(payload[t], dict):
                tool_name = t
                arguments = payload[t]
                break

    has_schema = bool(tool_name and isinstance(tool_name, str) and isinstance(arguments, dict))
    is_allowed = tool_name in allowed_tools if tool_name else False
    matches_req = (tool_name == required_tool) if (required_tool and tool_name) else True

    score = 0.0
    if has_schema and is_allowed and matches_req:
        score = 1.0
    elif has_schema and is_allowed:
        score = 0.8
    elif has_schema:
        score = 0.5
    else:
        score = 0.2  # Parsed JSON, but invalid tool call schema

    return {
        "is_valid_json": True,
        "has_valid_schema": has_schema,
        "tool_name": tool_name,
        "is_allowed_tool": is_allowed,
        "matches_required_tool": matches_req,
        "score": score,
        "error": None,
    }


# ==============================================================================
# 2. Grounded Synthesis & Citation Metrics
# ==============================================================================

# Matches DeepSearch format: [^chunk_id] or [^1] or [^chunk_001]
CITATION_REGEX = re.compile(r"\[\^([a-zA-Z0-9_-]+)\]")
# Matches non-standard bracket citation format like [1], [chunk_001] (for diagnosing base model)
BRACKET_REGEX = re.compile(r"\[([a-zA-Z0-9_-]+)\]")


def evaluate_grounded_synthesis(
    response_text: str,
    context_chunks: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate citation markers and ground truth chunk attribution.

    Checks:
    - Presence of strict [^chunk_id] markers
    - Precision: cited chunk IDs must exist in context_chunks (detects hallucinations)
    - Fallback: checks if model used non-standard [chunk_id] or [1] instead of [^chunk_id]
    """
    context_chunks_set = set(context_chunks or [])

    # Extract all [^chunk_id] citations
    citations = CITATION_REGEX.findall(response_text)
    citation_count = len(citations)

    # Check for legacy/unformatted bracket citations
    raw_bracket_matches = BRACKET_REGEX.findall(response_text)
    # Filter out markdown links or words like [Chunk]
    bracket_citations = [
        m for m in raw_bracket_matches if m in context_chunks_set or m.isdigit()
    ]

    valid_citations = [c for c in citations if c in context_chunks_set]
    hallucinated_citations = [c for c in citations if c not in context_chunks_set]

    # Citation precision: percentage of cited chunks that actually exist in the prompt
    if citation_count > 0:
        citation_precision = len(valid_citations) / citation_count
    else:
        citation_precision = 0.0

    # Chunk recall: percentage of available prompt chunks cited
    if context_chunks_set:
        chunk_coverage = len(set(valid_citations)) / len(context_chunks_set)
    else:
        chunk_coverage = 0.0

    word_count = max(len(response_text.split()), 1)
    citation_density = (citation_count / word_count) * 100.0  # citations per 100 words

    # Score calculation:
    # 0.5 for having valid [^chunk_id] syntax
    # + 0.3 for citation precision (no hallucinated chunk IDs)
    # + 0.2 for coverage of provided chunks
    score = 0.0
    if citation_count > 0:
        score += 0.5
        score += 0.3 * citation_precision
        score += 0.2 * min(chunk_coverage, 1.0)
    elif bracket_citations:
        # Partial credit if model cited in [chunk_id] format instead of [^chunk_id]
        score = 0.25

    return {
        "has_citations": citation_count > 0,
        "citation_count": citation_count,
        "valid_citations": valid_citations,
        "hallucinated_citations": hallucinated_citations,
        "citation_precision": round(citation_precision, 4),
        "chunk_coverage": round(chunk_coverage, 4),
        "citation_density": round(citation_density, 2),
        "used_standard_format": citation_count > 0,
        "used_bracket_fallback": len(bracket_citations) > 0 and citation_count == 0,
        "score": round(score, 4),
    }


# ==============================================================================
# 3. Query Decomposition Metrics
# ==============================================================================


def extract_queries_from_text(text: str) -> list[str]:
    """Extract individual search queries from model output."""
    queries: list[str] = []

    # 1. Check if model replied with JSON list: ["query 1", "query 2"]
    payload = extract_json_payload(text)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item.strip():
                queries.append(item.strip())
        if queries:
            return queries
    elif isinstance(payload, dict):
        for k in ["queries", "search_queries", "subqueries"]:
            if k in payload and isinstance(payload[k], list):
                for item in payload[k]:
                    if isinstance(item, str) and item.strip():
                        queries.append(item.strip())
                if queries:
                    return queries

    # 2. Line-by-line regex extraction: 1. "query" or - "query" or 1. query
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Pattern like: 1. "..." or 1. ... or - "..." or - ...
        m = re.match(r"^(?:\d+[\.\)]|[-*•])\s*(?:[\"']?)(.+?)(?:[\"']?)$", line)
        if m:
            candidate = m.group(1).strip()
            # Ignore headers like "Search Queries:"
            if len(candidate) > 5 and not candidate.endswith(":") and "query" not in candidate.lower()[:7]:
                queries.append(candidate.strip('"\'` '))

    # Fallback: if no list found, split by lines
    if not queries:
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 10 and not line.startswith("#"):
                queries.append(line.strip('"\'` '))

    return queries


def calculate_query_diversity(queries: list[str]) -> float:
    """Compute average pairwise word-Jaccard distance across queries (0.0=identical, 1.0=fully orthogonal)."""
    if len(queries) < 2:
        return 1.0 if queries else 0.0

    word_sets = [set(re.findall(r"\b\w+\b", q.lower())) for q in queries]
    distances = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            s1, s2 = word_sets[i], word_sets[j]
            union = s1 | s2
            if not union:
                continue
            inter = s1 & s2
            jaccard_sim = len(inter) / len(union)
            distances.append(1.0 - jaccard_sim)

    return sum(distances) / len(distances) if distances else 0.0


def evaluate_query_decomposition(
    response_text: str,
    target_min: int = 3,
    target_max: int = 5,
) -> dict[str, Any]:
    """Evaluate query count, target range compliance, and query orthogonality."""
    queries = extract_queries_from_text(response_text)
    count = len(queries)
    is_in_range = target_min <= count <= target_max
    diversity = calculate_query_diversity(queries)

    # Score:
    # 0.5 for having between 3 and 5 queries
    # 0.5 for query orthogonality / diversity
    score = 0.0
    if is_in_range:
        score += 0.5
    elif count > 0:
        score += 0.25

    score += 0.5 * diversity

    return {
        "query_count": count,
        "is_in_target_range": is_in_range,
        "queries": queries,
        "diversity_score": round(diversity, 4),
        "score": round(score, 4),
    }


# ==============================================================================
# 4. Gap Reflection Metrics
# ==============================================================================


def evaluate_gap_reflection(
    response_text: str,
    expected_gaps: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate identification of missing facts and follow-up query formulation."""
    lower_text = response_text.lower()

    # Detect gap indicators
    gap_indicators = [
        "gap", "missing", "lacks", "omits", "does not cover", "not addressed",
        "insufficient", "unaddressed", "needed", "requires",
    ]
    has_gap_analysis = any(ind in lower_text for ind in gap_indicators)

    # Detect follow-up query indicators
    query_indicators = ["follow-up", "follow up", "query", "queries", "search for", "investigate"]
    has_followup_queries = any(ind in lower_text for ind in query_indicators)

    # Extract follow-up queries if present
    extracted_queries = extract_queries_from_text(response_text)

    # Check keyword overlap with expected gaps
    expected_overlap = 0.0
    if expected_gaps:
        matched = 0
        for gap in expected_gaps:
            words = [w for w in re.findall(r"\b\w{4,}\b", gap.lower())]
            if any(w in lower_text for w in words):
                matched += 1
        expected_overlap = matched / len(expected_gaps)

    # Score:
    score = 0.0
    if has_gap_analysis:
        score += 0.4
    if has_followup_queries or len(extracted_queries) >= 2:
        score += 0.3
    score += 0.3 * expected_overlap

    return {
        "has_gap_analysis": has_gap_analysis,
        "has_followup_queries": has_followup_queries,
        "extracted_followup_count": len(extracted_queries),
        "expected_aspect_overlap": round(expected_overlap, 4),
        "score": round(score, 4),
    }
