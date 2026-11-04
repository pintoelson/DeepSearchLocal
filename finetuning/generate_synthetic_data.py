"""Synthetic trajectory generation engine via OpenRouter frontier models.

DeepSearch-Local requires instruction-tuning trajectories across 4 agent tasks:
1. Query Decomposition (25%): Multi-query generation from complex research prompts.
2. Tool Calling & Error Recovery (25%): Valid JSON schema tool calls (web_search, fetch_page_markdown)
   and error-recovery / query reformulation trajectories.
3. Grounded Synthesis & Citations (35%): Multi-chunk evidence synthesis with inline [^chunk_id] citations.
4. Gap Detection & Reflection (15%): Evaluating drafts, identifying evidence gaps, and formulating follow-ups.

Optimized for OpenRouter free tier (50 requests/day quota) using multi-sample batching
(10-15 trajectories per prompt) to achieve ~300-500 trajectories/day at $0.00 cost.
Supports strict semantic quality filtering, atomic resume checkpointing, and offline dry-run mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

from finetuning.dataset import validate_dataset_schema

# Task constants
TASK_QUERY_DECOMPOSITION = "query_decomposition"
TASK_TOOL_CALLING = "tool_calling"
TASK_GROUNDED_SYNTHESIS = "grounded_synthesis"
TASK_GAP_REFLECTION = "gap_reflection"

TASK_DISTRIBUTION = {
    TASK_QUERY_DECOMPOSITION: 0.25,
    TASK_TOOL_CALLING: 0.25,
    TASK_GROUNDED_SYNTHESIS: 0.35,
    TASK_GAP_REFLECTION: 0.15,
}

SYSTEM_PROMPTS = {
    TASK_QUERY_DECOMPOSITION: (
        "You are DeepSearch-Local, an expert autonomous deep research agent. "
        "Decompose complex, ambiguous, or multi-faceted research topics into a set of "
        "highly targeted, orthogonal search queries optimized for search engines."
    ),
    TASK_TOOL_CALLING: (
        "You are DeepSearch-Local, an expert autonomous deep research agent. "
        "Execute search and scraping tools strictly using valid JSON schemas. "
        "When searches fail or return insufficient data, reformulate queries intelligently."
    ),
    TASK_GROUNDED_SYNTHESIS: (
        "You are DeepSearch-Local, an expert autonomous deep research agent. "
        "Synthesize factual, concise research sections grounded strictly in the provided "
        "retrieved evidence passages. Every factual claim must be attributed using inline citation "
        "markers matching the exact passage ID format: [^chunk_id]."
    ),
    TASK_GAP_REFLECTION: (
        "You are DeepSearch-Local, an expert autonomous deep research agent. "
        "Critically evaluate current section drafts against research outlines. "
        "Identify missing factual evidence, pinpoint sections needing revision, and "
        "formulate targeted follow-up queries."
    ),
}


# ==============================================================================
# Seed Topic Taxonomy & Sampler
# ==============================================================================


@dataclass
class TopicSeed:
    domain_id: str
    domain_name: str
    topic: str
    subtopics: list[str]


def load_seed_topics(config_path: str | Path = "./finetuning/configs/seed_topics.yaml") -> list[TopicSeed]:
    """Load research domains and topic seeds from YAML configuration."""
    path = Path(config_path)
    if not path.is_file():
        # Fallback minimal seeds if file is missing
        return [
            TopicSeed(
                domain_id="ai_systems",
                domain_name="AI Systems",
                topic="Current state of 1-bit LLMs (BitNet b1.58)",
                subtopics=["Ternary weights", "FP16 comparison", "Hardware acceleration"],
            ),
            TopicSeed(
                domain_id="energy_storage",
                domain_name="Energy Storage",
                topic="Advances in solid-state battery electrolytes",
                subtopics=["Sulfide electrolytes", "Lithium dendrites", "Pilot manufacturing"],
            ),
        ]

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    seeds: list[TopicSeed] = []
    domains = data.get("domains", {})
    for dom_id, dom_data in domains.items():
        name = dom_data.get("name", dom_id)
        for s in dom_data.get("seeds", []):
            seeds.append(
                TopicSeed(
                    domain_id=dom_id,
                    domain_name=name,
                    topic=s.get("topic", ""),
                    subtopics=s.get("subtopics", []),
                )
            )
    return seeds


# ==============================================================================
# OpenRouter API Client with Rate Limiting & Quota Management
# ==============================================================================


class OpenRouterClient:
    """Client for OpenRouter API with rate-limiting, exponential backoff, and quota awareness."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        max_retries: int = 5,
        timeout: int = 60,
    ):
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()

    def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Call OpenRouter API with exponential backoff on HTTP 429 and 5xx."""
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Please provide it via environment variable, "
                ".env file, or --api_key CLI flag."
            )

        target_model = model or self.default_model
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/deepsearch-local",
            "X-Title": "DeepSearch-Local",
        }
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        backoff = 3.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError(f"No choices returned by model: {data}")
                    content = choices[0].get("message", {}).get("content")
                    if not content:
                        raise ValueError(f"Empty content returned by model: {choices[0]}")
                    return content

                if response.status_code == 429:
                    err_text = response.text
                    print(
                        f"⚠️ Rate limited (HTTP 429) on attempt {attempt}/{self.max_retries}. "
                        f"Backing off for {backoff:.1f}s... Details: {err_text[:120]}"
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                if response.status_code in {500, 502, 503, 504}:
                    print(
                        f"⚠️ OpenRouter server error ({response.status_code}) on attempt {attempt}/{self.max_retries}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue

                # Non-retryable error
                raise RuntimeError(
                    f"OpenRouter API error (HTTP {response.status_code}): {response.text[:300]}"
                )

            except (requests.Timeout, requests.ConnectionError) as req_err:
                print(
                    f"⚠️ Network error on attempt {attempt}/{self.max_retries}: {req_err}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 1.5

        raise RuntimeError(f"Exceeded max retries ({self.max_retries}) calling OpenRouter model {target_model}")

    def check_quota(self) -> dict[str, Any]:
        """Check free model daily requests quota."""
        if not self.api_key:
            return {"error": "No API key"}
        url = f"{self.base_url}/auth/key"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            res = self.session.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json().get("data", {})
                return {
                    "is_free_tier": data.get("is_free_tier", False),
                    "free_requests": data.get("free_model_daily_requests", {}),
                    "usage": data.get("usage", 0.0),
                }
        except Exception as e:
            return {"error": str(e)}
        return {"error": f"HTTP {res.status_code}"}


class GeminiClient:
    """Client for Google AI Studio (Gemini) REST API with rate-limiting and backoff."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gemini-2.5-flash",
        max_retries: int = 5,
        timeout: int = 60,
    ):
        load_dotenv()
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or ""
            )
        self.default_model = default_model
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()

    def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Call Google AI Studio REST API."""
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set. Please provide it via environment "
                "variable, .env file, or --api_key CLI flag."
            )

        target_model = model or self.default_model
        if target_model.startswith("models/"):
            target_model = target_model[len("models/"):]
        elif target_model.startswith("google/"):
            target_model = target_model[len("google/"):]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"

        system_instruction_text = ""
        contents = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_instruction_text += content + "\n"
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system_instruction_text.strip():
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction_text.strip()}]
            }

        backoff = 3.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"No candidates returned by Gemini: {data}")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts:
                        raise ValueError(f"Empty parts returned by Gemini: {candidates[0]}")
                    return parts[0].get("text", "")

                if response.status_code == 429:
                    print(
                        f"⚠️ Rate limited (HTTP 429) on attempt {attempt}/{self.max_retries}. "
                        f"Backing off for {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                if response.status_code in {500, 503, 504}:
                    print(
                        f"⚠️ Google API server error ({response.status_code}) on attempt {attempt}/{self.max_retries}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue

                raise RuntimeError(
                    f"Google AI Studio API error (HTTP {response.status_code}): {response.text[:300]}"
                )

            except (requests.Timeout, requests.ConnectionError) as req_err:
                print(
                    f"⚠️ Network error on attempt {attempt}/{self.max_retries}: {req_err}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 1.5

        raise RuntimeError(f"Exceeded max retries calling Gemini model {target_model}")

    def check_quota(self) -> dict[str, Any]:
        """Report Google AI Studio free tier limits."""
        return {
            "provider": "google_ai_studio",
            "free_requests": {"limit": 1500, "rate": "15 RPM", "daily": "1500 RPD"},
        }


class RWTHGPTClient:
    """Client for RWTH-GPT / KI:connect OpenAI-compatible API with dynamic model routing."""

    UNLIMITED_MODELS = [
        "gpt-oss-120b",
        "mistralai-mistral-small-4-119b",
    ]
    MODEL_LUNA = "gpt-6-luna"
    MODEL_SOL = "gpt-6-sol"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-oss-120b",
        max_retries: int = 5,
        timeout: int = 60,
        luna_budget: int = 50,
        sol_budget: int = 20,
    ):
        load_dotenv()
        self.api_key = (
            api_key
            or os.environ.get("RWTH_GPT_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.environ.get("RWTH_GPT_BASE_URL")
            or "https://chat.kiconnect.nrw/api/v1"
        ).rstrip("/")
        self.default_model = default_model
        self.max_retries = max_retries
        self.timeout = timeout
        self.luna_budget = min(luna_budget, 250)
        self.sol_budget = min(sol_budget, 60)
        self.session = requests.Session()

        self.request_counts: dict[str, int] = {
            m: 0 for m in self.UNLIMITED_MODELS + [self.MODEL_LUNA, self.MODEL_SOL]
        }
        self.unlimited_round_robin_idx: int = 0
        self.exhausted_models: set[str] = set()

    def select_model_for_task(self, task_type: str, strategy: str = "router") -> str:
        """Select model according to task type and quota availability."""
        if strategy == "single":
            return self.default_model

        # Specialized reasoning tasks: Grounded Synthesis & Gap Reflection
        if task_type in {TASK_GROUNDED_SYNTHESIS, TASK_GAP_REFLECTION}:
            if (
                self.request_counts.get(self.MODEL_LUNA, 0) < self.luna_budget
                and self.MODEL_LUNA not in self.exhausted_models
            ):
                return self.MODEL_LUNA
            if (
                self.request_counts.get(self.MODEL_SOL, 0) < self.sol_budget
                and self.MODEL_SOL not in self.exhausted_models
            ):
                return self.MODEL_SOL

        # Query decomposition, tool calling, or fallback: round-robin over unlimited models
        model = self.UNLIMITED_MODELS[self.unlimited_round_robin_idx % len(self.UNLIMITED_MODELS)]
        self.unlimited_round_robin_idx += 1
        return model

    def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Call RWTH-GPT OpenAI-compatible API with exponential backoff and quota fallback."""
        if not self.api_key:
            raise ValueError(
                "RWTH_GPT_API_KEY is not set. Please provide it via environment variable, "
                ".env file, or --api_key CLI flag."
            )

        target_model = model or self.default_model
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        backoff = 2.0
        for attempt in range(1, self.max_retries + 1):
            payload: dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "max_completion_tokens": max(max_tokens, 8192),
            }
            # Reasoning models (gpt-6-luna, gpt-6-sol) require reasoning_effort and default temperature (1.0)
            if target_model in {self.MODEL_LUNA, self.MODEL_SOL}:
                payload["reasoning_effort"] = "low"
            else:
                payload["temperature"] = temperature

            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError(f"No choices returned by RWTH-GPT: {data}")
                    content = choices[0].get("message", {}).get("content")
                    if not content:
                        raise ValueError(f"Empty content returned by RWTH-GPT: {choices[0]}")
                    self.request_counts[target_model] = self.request_counts.get(target_model, 0) + 1
                    return content

                if response.status_code == 429:
                    err_text = response.text
                    print(
                        f"⚠️ Rate limited / quota reached (HTTP 429) for '{target_model}'. "
                        f"Details: {err_text[:120]}"
                    )
                    # If quota-limited model hits 429, mark exhausted and switch to unlimited model
                    if target_model in {self.MODEL_LUNA, self.MODEL_SOL}:
                        print(f"⚠️ Marking {target_model} as exhausted. Falling back to unlimited pool.")
                        self.exhausted_models.add(target_model)
                        target_model = self.UNLIMITED_MODELS[
                            self.unlimited_round_robin_idx % len(self.UNLIMITED_MODELS)
                        ]
                        self.unlimited_round_robin_idx += 1
                        time.sleep(1.0)
                        continue

                    print(f"Backing off for {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                if response.status_code in {500, 502, 503, 504}:
                    print(
                        f"⚠️ RWTH-GPT server error ({response.status_code}) on attempt {attempt}/{self.max_retries}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue

                raise RuntimeError(
                    f"RWTH-GPT API error (HTTP {response.status_code}): {response.text[:300]}"
                )

            except (requests.Timeout, requests.ConnectionError) as req_err:
                print(
                    f"⚠️ Network error on attempt {attempt}/{self.max_retries}: {req_err}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 1.5

        raise RuntimeError(f"Exceeded max retries calling RWTH-GPT model {target_model}")

    def check_quota(self) -> dict[str, Any]:
        """Report model usage and quota limits."""
        return {
            "provider": "rwthgpt",
            "base_url": self.base_url,
            "request_counts": dict(self.request_counts),
            "limits": {
                self.MODEL_LUNA: {"budget": self.luna_budget, "hard_limit": 250},
                self.MODEL_SOL: {"budget": self.sol_budget, "hard_limit": 60},
                "unlimited_models": self.UNLIMITED_MODELS,
            },
        }


# ==============================================================================
# Multi-Stage Quality Gate & Semantic Validator
# ==============================================================================


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str | None = None
    task_type: str | None = None


def extract_citations(text: str) -> list[str]:
    """Extract all [^chunk_id] citation markers from text."""
    return re.findall(r"\[\^([a-zA-Z0-9_\-]+)\]", text)


def clean_json_payload(text: str) -> str:
    """Strip markdown code block fences (e.g. ```json ... ```) from JSON string."""
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def extract_chunk_ids_from_prompt(prompt_text: str) -> set[str]:
    """Extract defined chunk IDs from user context prompt."""
    # Matches patterns like [Chunk id]: or [id]:
    return set(re.findall(r"\[(?:Chunk\s+)?([a-zA-Z0-9_\-]+)\]:", prompt_text, re.IGNORECASE))


def validate_single_trajectory(record: dict[str, Any], expected_task: str | None = None) -> ValidationResult:
    """Perform deep semantic validation on a single ChatML trajectory."""
    if not isinstance(record, dict) or "messages" not in record:
        return ValidationResult(False, "Missing 'messages' key")

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return ValidationResult(False, "'messages' must contain at least user and assistant turns")

    user_msg: str | None = None
    assistant_msg: str | None = None

    for m in messages:
        if not isinstance(m, dict):
            return ValidationResult(False, "Message entry is not a dict")
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return ValidationResult(False, f"Message for role '{role}' has empty content")
        if role == "user":
            user_msg = content
        elif role == "assistant":
            assistant_msg = content

    if not user_msg or not assistant_msg:
        return ValidationResult(False, "Must contain both non-empty user and assistant messages")

    # Determine or use expected task type
    task = expected_task
    if not task:
        # Infer task from assistant content
        if '"queries"' in assistant_msg:
            task = TASK_QUERY_DECOMPOSITION
        elif '"tool"' in assistant_msg:
            task = TASK_TOOL_CALLING
        elif '"has_gaps"' in assistant_msg:
            task = TASK_GAP_REFLECTION
        elif "[^" in assistant_msg:
            task = TASK_GROUNDED_SYNTHESIS
        else:
            task = TASK_GROUNDED_SYNTHESIS

    # 1. Query Decomposition Validation
    if task == TASK_QUERY_DECOMPOSITION:
        try:
            parsed = json.loads(clean_json_payload(assistant_msg))
            if not isinstance(parsed, dict) or "queries" not in parsed:
                return ValidationResult(False, "Task 1 output must be a JSON object with 'queries' key", task)
            queries = parsed["queries"]
            if not isinstance(queries, list) or len(queries) < 2:
                return ValidationResult(False, "Task 1 'queries' must be a list with at least 2 queries", task)
            for q in queries:
                if not isinstance(q, str) or len(q.strip()) < 5:
                    return ValidationResult(False, "Task 1 query must be a string of at least 5 chars", task)
        except json.JSONDecodeError as e:
            return ValidationResult(False, f"Task 1 assistant output is not valid JSON: {e}", task)

    # 2. Tool Calling Validation
    elif task == TASK_TOOL_CALLING:
        try:
            parsed = json.loads(clean_json_payload(assistant_msg))
            if not isinstance(parsed, dict):
                return ValidationResult(False, "Task 2 output must be a JSON object", task)
            tool = parsed.get("tool")
            args = parsed.get("arguments")
            if tool not in {"web_search", "fetch_page_markdown"}:
                return ValidationResult(False, f"Task 2 invalid tool: '{tool}'", task)
            if not isinstance(args, dict):
                return ValidationResult(False, "Task 2 'arguments' must be a JSON object", task)
            if tool == "web_search":
                if "query" not in args or not isinstance(args["query"], str) or not args["query"].strip():
                    return ValidationResult(False, "Task 2 web_search requires non-empty 'query' string", task)
            elif tool == "fetch_page_markdown":
                if "url" not in args or not isinstance(args["url"], str) or not args["url"].startswith("http"):
                    return ValidationResult(False, "Task 2 fetch_page_markdown requires valid 'url' string", task)
        except json.JSONDecodeError as e:
            return ValidationResult(False, f"Task 2 assistant output is not valid JSON: {e}", task)

    # 3. Grounded Synthesis Validation
    elif task == TASK_GROUNDED_SYNTHESIS:
        citations = extract_citations(assistant_msg)
        if not citations:
            return ValidationResult(False, "Task 3 must contain at least one inline [^chunk_id] citation", task)
        available_chunks = extract_chunk_ids_from_prompt(user_msg)
        if available_chunks:
            for cite in citations:
                if cite not in available_chunks:
                    return ValidationResult(
                        False,
                        f"Task 3 cited hallucinated chunk '{cite}' not present in context chunks: {available_chunks}",
                        task,
                    )
        if len(assistant_msg.split()) < 15:
            return ValidationResult(False, "Task 3 synthesis too brief (<15 words)", task)

    # 4. Gap Reflection Validation
    elif task == TASK_GAP_REFLECTION:
        try:
            parsed = json.loads(clean_json_payload(assistant_msg))
            if not isinstance(parsed, dict):
                return ValidationResult(False, "Task 4 output must be a JSON object", task)
            if "has_gaps" not in parsed or not isinstance(parsed["has_gaps"], bool):
                return ValidationResult(False, "Task 4 requires boolean 'has_gaps'", task)
            if "sections_to_revise" not in parsed or not isinstance(parsed["sections_to_revise"], list):
                return ValidationResult(False, "Task 4 requires 'sections_to_revise' list", task)
            if "follow_up_queries" not in parsed or not isinstance(parsed["follow_up_queries"], list):
                return ValidationResult(False, "Task 4 requires 'follow_up_queries' list", task)
        except json.JSONDecodeError as e:
            return ValidationResult(False, f"Task 4 assistant output is not valid JSON: {e}", task)

    return ValidationResult(True, None, task)


# ==============================================================================
# Prompt Construction & Batch Parsing
# ==============================================================================


def build_batch_generation_prompt(
    task_type: str,
    seeds: list[TopicSeed],
    batch_size: int = 10,
    include_error_recovery: bool = False,
) -> list[dict[str, str]]:
    """Build a multi-sample batch generation prompt requesting a JSON array of trajectories."""
    sampled_topics = [f"- Topic: {s.topic} | Focus: {', '.join(s.subtopics[:2])}" for s in seeds[:batch_size]]
    topics_block = "\n".join(sampled_topics)

    system_instruction = (
        "You are an expert synthetic data generator producing high-quality instruction-tuning datasets "
        "for fine-tuning a 3-Billion parameter Small Language Model for an autonomous deep research agent. "
        "You must output ONLY a valid JSON array containing exactly "
        f"{batch_size} trajectory objects. Do NOT include markdown text outside the JSON array."
    )

    if task_type == TASK_QUERY_DECOMPOSITION:
        schema_example = json.dumps({
            "task": "query_decomposition",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS[TASK_QUERY_DECOMPOSITION]},
                {"role": "user", "content": "Generate targeted search queries to investigate: <Topic>"},
                {
                    "role": "assistant",
                    "content": json.dumps({"queries": ["query 1", "query 2", "query 3"]}, indent=2),
                },
            ],
        }, indent=2)

        task_instruction = (
            f"Generate {batch_size} diverse query decomposition trajectories based on these topics:\n"
            f"{topics_block}\n\n"
            "Each trajectory must follow this JSON schema:\n"
            f"{schema_example}\n\n"
            "Make each query specific, technical, and orthogonal (exploring different dimensions)."
        )

    elif task_type == TASK_TOOL_CALLING:
        error_clause = ""
        if include_error_recovery:
            error_clause = (
                "For approximately 20% of the trajectories, model an error-recovery scenario where a prior "
                "search returned 0 results or failed, and the assistant reformulates a broader or alternative query.\n\n"
            )

        schema_example = json.dumps({
            "task": "tool_calling",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS[TASK_TOOL_CALLING]},
                {"role": "user", "content": "Search for latest research on <Topic>"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"tool": "web_search", "arguments": {"query": "<search terms>", "max_results": 5}},
                        indent=2,
                    ),
                },
            ],
        }, indent=2)

        task_instruction = (
            f"Generate {batch_size} tool-calling trajectories based on these topics:\n"
            f"{topics_block}\n\n"
            f"{error_clause}"
            "DeepSearch-Local has two available tools:\n"
            "1. web_search(query: str, max_results: int = 5)\n"
            "2. fetch_page_markdown(url: str)\n\n"
            "Assistant response MUST be strictly valid JSON calling either web_search or fetch_page_markdown:\n"
            '{"tool": "web_search", "arguments": {"query": "...", "max_results": 5}}\n'
            "or\n"
            '{"tool": "fetch_page_markdown", "arguments": {"url": "https://..."}}\n\n'
            "Each trajectory must follow this JSON schema:\n"
            f"{schema_example}"
        )

    elif task_type == TASK_GROUNDED_SYNTHESIS:
        schema_example = json.dumps({
            "task": "grounded_synthesis",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS[TASK_GROUNDED_SYNTHESIS]},
                {"role": "user", "content": "Context:\n[Chunk chunk_a1]: ...\n[Chunk chunk_b2]: ...\n\nSummarize..."},
                {"role": "assistant", "content": "Synthesis text [^chunk_a1] further facts [^chunk_b2]."},
            ],
        }, indent=2)

        task_instruction = (
            f"Generate {batch_size} grounded synthesis trajectories based on these topics:\n"
            f"{topics_block}\n\n"
            "In each trajectory:\n"
            "1. The 'user' message provides a research query along with 2 to 3 simulated retrieved text chunks, "
            "formatted as '[Chunk chunk_xyz]: <passage text>'.\n"
            "2. The 'assistant' message synthesizes the facts into 1-2 coherent, technical paragraphs, "
            "citing sources inline using '[^chunk_xyz]' directly after every cited claim.\n"
            "IMPORTANT: Every single citation [^chunk_xyz] MUST correspond to a chunk present in the user context. "
            "Zero hallucinated citations.\n\n"
            "Each trajectory must follow this JSON schema:\n"
            f"{schema_example}"
        )

    else:  # TASK_GAP_REFLECTION
        schema_example = json.dumps({
            "task": "gap_reflection",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS[TASK_GAP_REFLECTION]},
                {"role": "user", "content": "Topic: ...\nOutline: ...\nCurrent Drafts: ..."},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "has_gaps": True,
                        "sections_to_revise": ["section_name"],
                        "follow_up_queries": ["query"],
                    }, indent=2),
                },
            ],
        }, indent=2)

        task_instruction = (
            f"Generate {batch_size} gap detection and reflection trajectories based on these topics:\n"
            f"{topics_block}\n\n"
            "In each trajectory:\n"
            "1. The 'user' message presents a research topic, an outline of required sections, and drafts of existing sections.\n"
            "2. The 'assistant' message evaluates whether critical evidence is missing and outputs a JSON object:\n"
            '{"has_gaps": true, "sections_to_revise": ["section_name"], "follow_up_queries": ["query"]}\n\n'
            "Each trajectory must follow this JSON schema:\n"
            f"{schema_example}"
        )

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": task_instruction},
    ]


def parse_batch_response(raw_response: str) -> list[dict[str, Any]]:
    """Extract and parse JSON array of trajectories from model completion."""
    cleaned = raw_response.strip()

    # 1. Try stripping markdown code blocks if present
    match = re.search(r"```(?:json)?\s*(\[\s*\{.*\}\s*\])\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Try finding the outermost brackets
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "trajectories" in parsed:
            return parsed["trajectories"]
    except json.JSONDecodeError:
        pass

    # 2. Robust progressive decoder: recovers all complete trajectories even if the overall array was cut off
    trajectories: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    pos = 0
    raw_len = len(raw_response)
    while pos < raw_len:
        idx = raw_response.find('"messages"', pos)
        if idx == -1:
            break
        # Find preceding '{' for this trajectory object
        start = raw_response.rfind("{", 0, idx)
        if start == -1:
            pos = idx + 10
            continue
        try:
            obj, end_offset = decoder.raw_decode(raw_response[start:])
            if isinstance(obj, dict) and "messages" in obj:
                trajectories.append(obj)
            pos = start + end_offset
        except Exception:
            pos = idx + 10

    if trajectories:
        return trajectories

    # 3. Fallback: line-by-line JSON parsing
    for line in raw_response.splitlines():
        line_str = line.strip().rstrip(",")
        if line_str.startswith("{") and line_str.endswith("}"):
            try:
                obj = json.loads(line_str)
                if "messages" in obj:
                    trajectories.append(obj)
            except Exception:
                continue
    return trajectories


# ==============================================================================
# Offline Dry-Run Synthesizer
# ==============================================================================


def generate_mock_trajectories(
    count: int,
    seeds: list[TopicSeed],
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    """Generate realistic mock trajectories offline without hitting OpenRouter API."""
    trajectories: list[dict[str, Any]] = []
    task_order = [
        TASK_QUERY_DECOMPOSITION,
        TASK_TOOL_CALLING,
        TASK_GROUNDED_SYNTHESIS,
        TASK_GAP_REFLECTION,
    ]

    for i in range(count):
        seed = seeds[i % len(seeds)]
        task = task_type if task_type else task_order[i % len(task_order)]
        topic = seed.topic
        sub1 = seed.subtopics[0] if seed.subtopics else "fundamental architecture"
        sub2 = seed.subtopics[1] if len(seed.subtopics) > 1 else "performance scaling"

        if task == TASK_QUERY_DECOMPOSITION:
            record = {
                "task": task,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPTS[TASK_QUERY_DECOMPOSITION],
                    },
                    {
                        "role": "user",
                        "content": f"Generate targeted search queries to investigate: {topic}.",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "queries": [
                                    f"{topic} {sub1} breakthrough analysis",
                                    f"{topic} {sub2} empirical benchmarks",
                                    f"{topic} industrial commercialization timeline 2025 2026",
                                ]
                            },
                            indent=2,
                        ),
                    },
                ],
            }

        elif task == TASK_TOOL_CALLING:
            is_error_recovery = (i % 5 == 0)
            if is_error_recovery:
                user_prompt = f"The search for '{topic} niche edgecase' returned 0 results. Reformulate the query."
                assistant_content = json.dumps(
                    {
                        "tool": "web_search",
                        "arguments": {
                            "query": f"{topic} {sub1} overview state of the art",
                            "max_results": 5,
                        },
                    },
                    indent=2,
                )
            else:
                user_prompt = f"Search for the latest research papers and data on {topic}."
                assistant_content = json.dumps(
                    {
                        "tool": "web_search",
                        "arguments": {
                            "query": f"{topic} {sub1} {sub2} research papers",
                            "max_results": 5,
                        },
                    },
                    indent=2,
                )

            record = {
                "task": task,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPTS[TASK_TOOL_CALLING],
                    },
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_content},
                ],
            }

        elif task == TASK_GROUNDED_SYNTHESIS:
            h1 = f"chunk_{hashlib.md5(f'{topic}_1_{i}'.encode()).hexdigest()[:6]}"
            h2 = f"chunk_{hashlib.md5(f'{topic}_2_{i}'.encode()).hexdigest()[:6]}"

            user_content = (
                f"Context:\n"
                f"[Chunk {h1}]: Recent investigations into {topic} demonstrate key advancements in {sub1}.\n"
                f"[Chunk {h2}]: Systematic evaluations show {sub2} enables substantial throughput gains.\n\n"
                f"Summarize the recent findings on {topic}."
            )
            assistant_content = (
                f"Recent experimental analyses of {topic} reveal significant progress in {sub1} [^{h1}]. "
                f"Furthermore, rigorous benchmarks confirm that {sub2} delivers considerable practical gains [^{h2}]."
            )

            record = {
                "task": task,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPTS[TASK_GROUNDED_SYNTHESIS],
                    },
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
            }

        else:  # TASK_GAP_REFLECTION
            record = {
                "task": task,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPTS[TASK_GAP_REFLECTION],
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Topic: {topic}\n"
                            f"Outline: 1. Overview 2. {sub1} 3. {sub2}\n"
                            f"Current Drafts: Section 1 and Section 2 drafted. Section 3 ({sub2}) lacks concrete metrics."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "has_gaps": True,
                                "sections_to_revise": [f"{sub2.lower().replace(' ', '_')}_metrics"],
                                "follow_up_queries": [f"{topic} {sub2} quantitative benchmarks dataset"],
                            },
                            indent=2,
                        ),
                    },
                ],
            }

        trajectories.append(record)

    return trajectories


# ==============================================================================
# Checkpointing and Execution Controller
# ==============================================================================


@dataclass
class GenerationCheckpoint:
    total_valid_samples: int = 0
    task_counts: dict[str, int] = field(
        default_factory=lambda: {
            TASK_QUERY_DECOMPOSITION: 0,
            TASK_TOOL_CALLING: 0,
            TASK_GROUNDED_SYNTHESIS: 0,
            TASK_GAP_REFLECTION: 0,
        }
    )
    seen_hashes: list[str] = field(default_factory=list)
    last_seed_idx: int = 0

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> "GenerationCheckpoint":
        path = Path(checkpoint_path)
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return cls(
                        total_valid_samples=data.get("total_valid_samples", 0),
                        task_counts=data.get("task_counts", {}),
                        seen_hashes=data.get("seen_hashes", []),
                        last_seed_idx=data.get("last_seed_idx", 0),
                    )
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint ({e}). Starting fresh.")
        return cls()

    def save(self, checkpoint_path: str | Path):
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def get_trajectory_hash(record: dict[str, Any]) -> str:
    """Generate deterministic hash for trajectory deduplication."""
    messages = record.get("messages", [])
    content_str = "|".join([f"{m.get('role')}:{m.get('content')}" for m in messages])
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()


def generate_trajectories(
    output_path: str = "./data/synthetic_dataset.jsonl",
    count: int = 1000,
    batch_size: int = 10,
    provider: str = "openrouter",
    model: str | None = None,
    model_strategy: str = "router",
    topics_config: str = "./finetuning/configs/seed_topics.yaml",
    api_key: str | None = None,
    dry_run: bool = False,
    checkpoint_path: str | None = None,
) -> Path:
    """End-to-end synthetic trajectory generation loop with batching, validation, and resume."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ckpt_file = (
        Path(checkpoint_path)
        if checkpoint_path
        else out_file.parent / f".{out_file.stem}_checkpoint.json"
    )

    tasks = [
        TASK_QUERY_DECOMPOSITION,
        TASK_TOOL_CALLING,
        TASK_GROUNDED_SYNTHESIS,
        TASK_GAP_REFLECTION,
    ]

    # Determine default model by provider
    target_model = model
    if not target_model:
        if provider == "gemini":
            target_model = "gemini-2.5-flash"
        elif provider == "rwthgpt":
            target_model = "router" if model_strategy == "router" else "gpt-oss-120b"
        else:
            target_model = "nvidia/nemotron-3-ultra-550b-a55b:free"

    print("=" * 80)
    print("🚀 DeepSearch-Local: Synthetic Trajectory Generation Engine")
    print("=" * 80)
    print(f"📁 Output Target:     {out_file.resolve()}")
    print(f"🎯 Target Samples:    {count}")
    print(f"📦 Batch Size:        {batch_size} samples/call")
    print(f"🌐 Provider:          {provider.upper()}")
    print(f"🤖 Generator Model:   {target_model} (Mode: {'DRY RUN (Offline)' if dry_run else 'Live API'})")
    print(f"💾 Checkpoint File:   {ckpt_file.name}")

    # Load topic taxonomy
    seeds = load_seed_topics(topics_config)
    print(f"📚 Loaded {len(seeds)} topic seeds across multi-domain taxonomy.")

    # Load checkpoint
    checkpoint = GenerationCheckpoint.load(ckpt_file)
    seen_hashes_set = set(checkpoint.seen_hashes)

    # If output file already exists, align sample count; if missing, reset checkpoint
    if out_file.is_file():
        is_val, err, actual_count = validate_dataset_schema(out_file)
        if is_val:
            checkpoint.total_valid_samples = actual_count
            print(f"ℹ️ Found existing dataset with {actual_count} valid trajectories. Resuming...")
            # If checkpoint seen_hashes count doesn't match actual_count, resync
            if len(checkpoint.seen_hashes) != actual_count:
                print("🔄 Re-syncing checkpoint hashes and task counts with dataset...")
                checkpoint.seen_hashes = []
                checkpoint.task_counts = {t: 0 for t in tasks}
                with out_file.open("r", encoding="utf-8") as f_in:
                    for line in f_in:
                        if line.strip():
                            rec = json.loads(line)
                            h = get_trajectory_hash(rec)
                            checkpoint.seen_hashes.append(h)
                            val_res = validate_single_trajectory(rec)
                            task_name = val_res.task_type or "grounded_synthesis"
                            checkpoint.task_counts[task_name] = checkpoint.task_counts.get(task_name, 0) + 1
                seen_hashes_set = set(checkpoint.seen_hashes)
                checkpoint.save(ckpt_file)
    else:
        checkpoint = GenerationCheckpoint()
        seen_hashes_set = set()

    remaining = count - checkpoint.total_valid_samples
    if remaining <= 0:
        print(f"✅ Target count of {count} already met in {out_file}. Generation complete!")
        return out_file

    print(f"⏳ Generating {remaining} additional trajectories to reach target of {count}...")

    # Initialize client if live run
    client: OpenRouterClient | GeminiClient | RWTHGPTClient | None = None
    if not dry_run:
        if provider == "gemini":
            client = GeminiClient(api_key=api_key, default_model=target_model)
            print("📊 Provider Info: Google AI Studio (15 RPM / 1,500 RPD free tier)")
        elif provider == "rwthgpt":
            client = RWTHGPTClient(
                api_key=api_key,
                default_model="gpt-oss-120b" if target_model == "router" else target_model,
            )
            q_info = client.check_quota()
            print(f"📊 Provider Info: RWTH-GPT ({q_info['base_url']})")
            print(f"   • Unlimited Models: {q_info['limits']['unlimited_models']}")
            print(f"   • Daily Limits: Luna (250/day), Sol (60/day)")
        else:
            client = OpenRouterClient(api_key=api_key, default_model=target_model)
            quota_info = client.check_quota()
            if "free_requests" in quota_info:
                reqs = quota_info["free_requests"]
                print(
                    f"📊 OpenRouter Free Quota: {reqs.get('remaining', 'N/A')} remaining "
                    f"out of {reqs.get('limit', 'N/A')} daily requests."
                )

    seed_idx = checkpoint.last_seed_idx

    # Main generation loop
    with out_file.open("a", encoding="utf-8") as f_out:
        while checkpoint.total_valid_samples < count:
            # Pick task based on current distribution balance
            current_counts = checkpoint.task_counts
            total_current = sum(current_counts.values()) or 1
            task_deficits = {
                t: TASK_DISTRIBUTION[t] - (current_counts[t] / total_current)
                for t in tasks
            }
            # Pick task with highest deficit
            target_task = max(task_deficits, key=task_deficits.get)  # type: ignore

            # Select seeds for this batch
            batch_seeds = [seeds[(seed_idx + k) % len(seeds)] for k in range(batch_size)]
            seed_idx = (seed_idx + batch_size) % len(seeds)

            # Determine model for this batch
            batch_model = target_model
            if provider == "rwthgpt" and isinstance(client, RWTHGPTClient) and model_strategy == "router":
                batch_model = client.select_model_for_task(target_task, strategy=model_strategy)

            print(
                f"\n🔄 Generating batch for task: [{target_task}] "
                f"via [{batch_model}] "
                f"(Progress: {checkpoint.total_valid_samples}/{count})"
            )

            if dry_run:
                raw_trajectories = generate_mock_trajectories(batch_size, batch_seeds, task_type=target_task)
            else:
                prompt_messages = build_batch_generation_prompt(
                    task_type=target_task,
                    seeds=batch_seeds,
                    batch_size=batch_size,
                    include_error_recovery=True,
                )
                try:
                    assert client is not None
                    completion = client.generate_chat_completion(
                        messages=prompt_messages,
                        model=batch_model,
                    )
                    raw_trajectories = parse_batch_response(completion)
                except Exception as ex:
                    print(f"❌ Batch generation failed: {ex}")
                    time.sleep(5)
                    continue

            # Validate each trajectory through quality gate
            accepted_in_batch = 0
            for item in raw_trajectories:
                # Format to standard messages list
                if not isinstance(item, dict) or "messages" not in item:
                    continue

                if target_task in {TASK_QUERY_DECOMPOSITION, TASK_TOOL_CALLING, TASK_GAP_REFLECTION}:
                    clean_record = {
                        "messages": [
                            m if m.get("role") != "assistant"
                            else {"role": "assistant", "content": clean_json_payload(m.get("content", ""))}
                            for m in item["messages"]
                        ]
                    }
                else:
                    clean_record = {"messages": item["messages"]}

                val_result = validate_single_trajectory(clean_record, expected_task=target_task)

                if not val_result.is_valid:
                    print(f"   ⚠️ Discarded sample: {val_result.error_message}")
                    continue

                t_hash = get_trajectory_hash(clean_record)
                if t_hash in seen_hashes_set:
                    print("   ⚠️ Discarded duplicate sample")
                    continue

                # Write atomically to file
                f_out.write(json.dumps(clean_record) + "\n")
                f_out.flush()

                seen_hashes_set.add(t_hash)
                checkpoint.seen_hashes.append(t_hash)
                checkpoint.total_valid_samples += 1
                task_key = val_result.task_type or target_task
                checkpoint.task_counts[task_key] = checkpoint.task_counts.get(task_key, 0) + 1
                accepted_in_batch += 1

                if checkpoint.total_valid_samples >= count:
                    break

            checkpoint.last_seed_idx = seed_idx
            checkpoint.save(ckpt_file)
            print(
                f"   ✅ Saved {accepted_in_batch}/{len(raw_trajectories)} valid trajectories. "
                f"Total: {checkpoint.total_valid_samples}/{count}"
            )

            # Politeness delay between batches
            if not dry_run:
                delay = 1.0 if provider == "rwthgpt" else (4.0 if provider == "gemini" else 2.0)
                time.sleep(delay)

    print("\n" + "=" * 80)
    print(f"🎉 Dataset generation complete! Total samples: {checkpoint.total_valid_samples}")
    print_dataset_statistics(out_file)
    print("=" * 80)
    return out_file


def print_dataset_statistics(dataset_path: str | Path):
    """Print category breakdown and statistics of a generated dataset."""
    path = Path(dataset_path)
    if not path.is_file():
        print(f"Dataset not found: {path}")
        return

    is_valid, err, total_count = validate_dataset_schema(path)
    if not is_valid:
        print(f"❌ Schema validation failed: {err}")
        return

    counts: dict[str, int] = {
        TASK_QUERY_DECOMPOSITION: 0,
        TASK_TOOL_CALLING: 0,
        TASK_GROUNDED_SYNTHESIS: 0,
        TASK_GAP_REFLECTION: 0,
        "other": 0,
    }
    total_assistant_words = 0
    total_citations = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assistant_text = ""
            for m in rec["messages"]:
                if m.get("role") == "assistant":
                    assistant_text = m.get("content", "")

            words = len(assistant_text.split())
            total_assistant_words += words
            cites = extract_citations(assistant_text)
            total_citations += len(cites)

            if '"queries"' in assistant_text:
                counts[TASK_QUERY_DECOMPOSITION] += 1
            elif '"tool"' in assistant_text:
                counts[TASK_TOOL_CALLING] += 1
            elif '"has_gaps"' in assistant_text:
                counts[TASK_GAP_REFLECTION] += 1
            elif "[^" in assistant_text:
                counts[TASK_GROUNDED_SYNTHESIS] += 1
            else:
                counts["other"] += 1

    print("\n📊 Dataset Quality & Breakdown Report:")
    print(f"   • Total Trajectories:      {total_count}")
    print(f"   • Average Assistant Words: {total_assistant_words / max(total_count, 1):.1f} words/sample")
    print(f"   • Total Grounded Citations:{total_citations}")
    print(f"   • Task Distribution:")
    for task_name, task_c in counts.items():
        pct = (task_c / max(total_count, 1)) * 100
        print(f"     - {task_name:<25}: {task_c:>4} ({pct:5.1f}%)")


def merge_datasets(input_files: list[str | Path], output_path: str | Path) -> Path:
    """Merge multiple JSONL trajectory files into one with deduplication and quality validation."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    valid_count = 0
    duplicate_count = 0
    invalid_count = 0

    temp_out = out_file.with_suffix(".merge_tmp")
    with temp_out.open("w", encoding="utf-8") as f_out:
        for in_path in input_files:
            p = Path(in_path)
            if not p.is_file():
                print(f"⚠️ Input file not found (skipping): {p}")
                continue
            print(f"📖 Merging records from: {p.name}...")
            with p.open("r", encoding="utf-8") as f_in:
                for line in f_in:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        rec = json.loads(line_str)
                    except Exception:
                        invalid_count += 1
                        continue
                    clean_rec = {"messages": rec.get("messages", [])}
                    val = validate_single_trajectory(clean_rec)
                    if not val.is_valid:
                        invalid_count += 1
                        continue
                    t_hash = get_trajectory_hash(clean_rec)
                    if t_hash in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(t_hash)
                    f_out.write(json.dumps(clean_rec) + "\n")
                    valid_count += 1

    temp_out.replace(out_file)
    print("\n" + "=" * 80)
    print(f"✅ Merge complete! Saved {valid_count} unique valid trajectories to: {out_file.resolve()}")
    if duplicate_count > 0:
        print(f"   ℹ️ Deduplicated: {duplicate_count} duplicate records removed.")
    if invalid_count > 0:
        print(f"   ℹ️ Filtered:     {invalid_count} malformed records removed.")
    print("=" * 80)
    print_dataset_statistics(out_file)
    return out_file


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="DeepSearch-Local Synthetic Trajectory Generator")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/synthetic_dataset.jsonl",
        help="Target output JSONL path",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        default=None,
        help="Merge multiple JSONL dataset files with deduplication into --output_path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Total trajectories to generate (default: 1000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Samples per generation call (default: 10)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        choices=["openrouter", "gemini", "rwthgpt"],
        help="Inference provider: 'openrouter', 'gemini' (Google AI Studio), or 'rwthgpt' (RWTH-GPT)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (defaults to provider's recommended free model or router)",
    )
    parser.add_argument(
        "--model_strategy",
        type=str,
        default="router",
        choices=["router", "single"],
        help="Model strategy for RWTH-GPT: 'router' (dynamic round-robin & task specialization) or 'single'",
    )
    parser.add_argument(
        "--topics_config",
        type=str,
        default="./finetuning/configs/seed_topics.yaml",
        help="Path to YAML seed topics",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API Key (defaults to OPENROUTER_API_KEY, GEMINI_API_KEY, or RWTH_GPT_API_KEY)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run offline mock generation without calling external APIs",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Verify and report statistics on an existing dataset file",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset distribution and quality stats",
    )

    args = parser.parse_args()

    if args.merge:
        merge_datasets(args.merge, args.output_path)
        return

    if args.verify_only or args.stats:
        print_dataset_statistics(args.output_path)
        return

    generate_trajectories(
        output_path=args.output_path,
        count=args.count,
        batch_size=args.batch_size,
        provider=args.provider,
        model=args.model,
        model_strategy=args.model_strategy,
        topics_config=args.topics_config,
        api_key=args.api_key,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
