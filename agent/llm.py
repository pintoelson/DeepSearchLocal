"""Unified LLM client abstraction supporting Ollama, OpenAI-compatible APIs, and Mock testing."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class StructuredOutputParser:
    """Robust extractor and sanitizer for JSON payloads embedded in LLM text completions."""

    @staticmethod
    def extract_json(raw_text: str) -> dict[str, Any] | list[Any]:
        """Extract and parse a JSON dictionary or list from raw model text.

        Handles:
        - Markdown fenced code blocks: ```json ... ``` or ``` ... ```
        - Leading/trailing conversational conversational filler
        - Trailing commas before closing braces
        """
        text = raw_text.strip()

        # 1. Direct parse attempt
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Markdown fence extraction (```json ... ``` or ``` ... ```)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                text = candidate

        # 3. Bracket extraction: find first '{' or '[' to matching last '}' or ']'
        dict_start = text.find("{")
        list_start = text.find("[")

        if dict_start != -1 and (list_start == -1 or dict_start < list_start):
            dict_end = text.rfind("}")
            if dict_end > dict_start:
                candidate = text[dict_start : dict_end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Sanitize trailing commas: ", \n }" or ", }"
                    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass
        elif list_start != -1:
            list_end = text.rfind("]")
            if list_end > list_start:
                candidate = text[list_start : list_end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass

        raise ValueError(f"Could not parse valid JSON from text: {raw_text[:200]}...")


class BaseAgentLLM(ABC):
    """Abstract interface for agent LLM execution."""

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Asynchronously generate a completion response from ChatML messages."""

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Synchronously generate a completion response."""
        import asyncio

        return asyncio.run(
            self.agenerate(messages, temperature=temperature, max_tokens=max_tokens)
        )


class AgentLLM(BaseAgentLLM):
    """Unified LLM client communicating with Ollama REST API or OpenAI-compatible endpoints."""

    def __init__(
        self,
        model: str = "deepsearch-3b:latest",
        provider: str = "ollama",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.provider = provider.lower()
        self.api_key = api_key or ""
        self.timeout = timeout

        if base_url:
            self.base_url = base_url.rstrip("/")
        elif self.provider == "ollama":
            self.base_url = "http://localhost:11434"
        else:
            self.base_url = "https://api.openai.com/v1"

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Query LLM endpoint asynchronously."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if self.provider == "ollama":
                url = f"{self.base_url}/api/chat"
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stop": ["<|im_end|>", "<|endoftext|>"],
                    },
                }
                headers = {"Content-Type": "application/json"}
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")

            # OpenAI-compatible (/v1/chat/completions)
            url = f"{self.base_url}/chat/completions" if "/v1" in self.base_url or not self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions"
            if not url.endswith("/chat/completions"):
                url = f"{self.base_url}/chat/completions"

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class MockAgentLLM(BaseAgentLLM):
    """Deterministic Mock LLM for offline testing and continuous integration."""

    def __init__(
        self,
        default_response: str = "Mock response",
        custom_handler: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self.default_response = default_response
        self.custom_handler = custom_handler
        self.call_history: list[list[dict[str, str]]] = []

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        self.call_history.append(messages)
        if self.custom_handler:
            return self.custom_handler(messages)
        return self.default_response
