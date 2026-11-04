"""Dual-engine LLM client wrapper supporting Ollama and vLLM via OpenAI-compatible API."""

from typing import Any, AsyncIterator, Dict, List, Optional


class LocalLLMClient:
    """Unified client for interacting with locally hosted Ollama or vLLM instances."""

    def __init__(self, provider: str = "ollama", base_url: str = "http://localhost:11434") -> None:
        self.provider = provider
        self.base_url = base_url

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to local LLM backend."""
        raise NotImplementedError("LLM client implementation in Phase 5.")

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """Stream token responses via SSE."""
        raise NotImplementedError("LLM client streaming implementation in Phase 5.")
        yield ""
