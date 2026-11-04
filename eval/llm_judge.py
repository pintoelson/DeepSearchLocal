"""LLM-as-a-Judge module supporting RWTH-GPT primary with Gemini & OpenRouter fallback."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
import requests
from dotenv import load_dotenv

load_dotenv()


class BaseJudge:
    """Abstract base judge interface."""

    def evaluate_pair(
        self,
        task: str,
        topic: str,
        user_prompt: str,
        response_base: str,
        response_ft: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockJudge(BaseJudge):
    """Offline heuristic judge used when APIs are unreachable or for fast dry runs."""

    def evaluate_pair(
        self,
        task: str,
        topic: str,
        user_prompt: str,
        response_base: str,
        response_ft: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Fast rule-based heuristic for offline verification
        has_citations_ft = "[^" in response_ft
        has_citations_base = "[^" in response_base

        if task == "grounded_synthesis":
            winner = "finetuned" if (has_citations_ft and not has_citations_base) else "tie"
        elif task == "tool_calling":
            winner = "finetuned" if ('"tool"' in response_ft and '"tool"' not in response_base) else "tie"
        else:
            winner = "tie"

        return {
            "judge_model": "mock-heuristic",
            "base_score": 3.0,
            "ft_score": 4.5 if winner == "finetuned" else 3.5,
            "winner": winner,
            "reasoning": f"Offline heuristic judge evaluated task '{task}'.",
        }


class RWTHGPTJudge(BaseJudge):
    """RWTH-GPT judge querying the KI-Connect NRW endpoint."""

    DEFAULT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
    UNLIMITED_MODELS = [
        "mistralai-mistral-small-4-119b",
        "gpt-oss-120b",
    ]
    MODEL_LUNA = "gpt-6-luna"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("RWTH_GPT_API_KEY", "")
        self.base_url = (base_url or os.environ.get("RWTH_GPT_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.MODEL_LUNA
        self.session = requests.Session()

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 8)

    def evaluate_pair(
        self,
        task: str,
        topic: str,
        user_prompt: str,
        response_base: str,
        response_ft: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("RWTH_GPT_API_KEY is not configured")

        system_instruction = (
            "You are an expert impartial AI evaluator judging model responses for an autonomous deep research agent.\n"
            "Compare Model A (Base SLM) vs Model B (Fine-Tuned SLM).\n"
            "Evaluate strictly on the criteria for the specified task:\n"
            "- tool_calling: strict JSON schema conformity, valid tool arguments, no extraneous chatter.\n"
            "- grounded_synthesis: strict adherence to provided context, zero extrapolation, correct inline [^chunk_id] citations.\n"
            "- query_decomposition: 3-5 orthogonal, specific search queries.\n"
            "- gap_reflection: clear identification of missing facts and actionable follow-up queries.\n\n"
            "Respond ONLY with a JSON object matching this schema:\n"
            "{\n"
            '  "base_score": <float 1.0 to 5.0>,\n'
            '  "ft_score": <float 1.0 to 5.0>,\n'
            '  "winner": "<base|finetuned|tie>",\n'
            '  "reasoning": "<concise 2-sentence rationale>"\n'
            "}"
        )

        user_content = (
            f"### Task: {task}\n"
            f"### Topic: {topic}\n"
            f"### User Prompt & Context:\n{user_prompt}\n\n"
            f"### Model A (Base SLM) Response:\n{response_base}\n\n"
            f"### Model B (Fine-Tuned SLM) Response:\n{response_ft}\n"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ]

        target_model = self.model
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "max_completion_tokens": 1024,
        }
        if target_model in {self.MODEL_LUNA, "gpt-6-sol"}:
            payload["reasoning_effort"] = "low"
        else:
            payload["temperature"] = 0.1

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Send request with retry
        for attempt in range(1, 3):
            try:
                res = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=45,
                )
                if res.status_code == 200:
                    data = res.json()
                    raw_text = data["choices"][0]["message"]["content"]
                    parsed = _parse_judge_json(raw_text)
                    parsed["judge_model"] = f"rwth-gpt/{target_model}"
                    return parsed
                elif res.status_code == 429 and target_model == self.MODEL_LUNA:
                    # Switch to unlimited model on rate limit
                    target_model = self.UNLIMITED_MODELS[0]
                    payload["model"] = target_model
                    payload.pop("reasoning_effort", None)
                    payload["temperature"] = 0.1
                    continue
                else:
                    raise RuntimeError(f"RWTH-GPT error HTTP {res.status_code}: {res.text[:200]}")
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(1.5)

        raise RuntimeError("RWTH-GPT request failed")


class GeminiJudge(BaseJudge):
    """Google Gemini REST API fallback judge."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        self.model = model
        self.session = requests.Session()

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 8)

    def evaluate_pair(
        self,
        task: str,
        topic: str,
        user_prompt: str,
        response_base: str,
        response_ft: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        prompt = (
            f"You are an expert impartial AI evaluator judging model responses for an autonomous deep research agent.\n"
            f"Compare Model A (Base SLM) vs Model B (Fine-Tuned SLM).\n\n"
            f"Task: {task}\n"
            f"Topic: {topic}\n"
            f"Input Prompt:\n{user_prompt}\n\n"
            f"Model A Response:\n{response_base}\n\n"
            f"Model B Response:\n{response_ft}\n\n"
            f"Respond ONLY with a JSON object in this format:\n"
            f'{{"base_score": <1.0-5.0>, "ft_score": <1.0-5.0>, "winner": "<base|finetuned|tie>", "reasoning": "<short rationale>"}}'
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        res = self.session.post(url, json=payload, timeout=30)
        if res.status_code != 200:
            raise RuntimeError(f"Gemini API error HTTP {res.status_code}: {res.text[:200]}")

        data = res.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = _parse_judge_json(raw_text)
        parsed["judge_model"] = f"gemini/{self.model}"
        return parsed


class OpenRouterJudge(BaseJudge):
    """OpenRouter API fallback judge."""

    def __init__(self, api_key: str | None = None, model: str = "meta-llama/llama-3.3-70b-instruct"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.session = requests.Session()

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 8)

    def evaluate_pair(
        self,
        task: str,
        topic: str,
        user_prompt: str,
        response_base: str,
        response_ft: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        prompt = (
            f"Task: {task}\nTopic: {topic}\nPrompt: {user_prompt}\n\n"
            f"Model A (Base): {response_base}\n\nModel B (Fine-Tuned): {response_ft}\n\n"
            f"Return JSON strictly: {{\"base_score\": <1.0-5.0>, \"ft_score\": <1.0-5.0>, \"winner\": \"<base|finetuned|tie>\", \"reasoning\": \"<rationale>\"}}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert AI evaluator. Reply strictly in JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        res = self.session.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code != 200:
            raise RuntimeError(f"OpenRouter error HTTP {res.status_code}: {res.text[:200]}")

        data = res.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = _parse_judge_json(raw_text)
        parsed["judge_model"] = f"openrouter/{self.model}"
        return parsed


class FallbackJudge(BaseJudge):
    """Orchestrates judgment: RWTH-GPT -> Gemini -> OpenRouter -> MockJudge."""

    def __init__(self, force_mock: bool = False):
        self.force_mock = force_mock
        self.rwth_judge = RWTHGPTJudge()
        self.gemini_judge = GeminiJudge()
        self.openrouter_judge = OpenRouterJudge()
        self.mock_judge = MockJudge()

    def evaluate_pair(
        self,
        task: str,
        topic: str,
        user_prompt: str,
        response_base: str,
        response_ft: str,
        context_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.force_mock:
            return self.mock_judge.evaluate_pair(task, topic, user_prompt, response_base, response_ft, context_data)

        # 1. Try RWTH-GPT (Primary)
        if self.rwth_judge.is_available():
            try:
                return self.rwth_judge.evaluate_pair(task, topic, user_prompt, response_base, response_ft, context_data)
            except Exception as e:
                print(f"⚠️ RWTH-GPT judge failed ({e}). Attempting Gemini fallback...")

        # 2. Try Gemini (Fallback 1)
        if self.gemini_judge.is_available():
            try:
                return self.gemini_judge.evaluate_pair(task, topic, user_prompt, response_base, response_ft, context_data)
            except Exception as e:
                print(f"⚠️ Gemini judge failed ({e}). Attempting OpenRouter fallback...")

        # 3. Try OpenRouter (Fallback 2)
        if self.openrouter_judge.is_available():
            try:
                return self.openrouter_judge.evaluate_pair(task, topic, user_prompt, response_base, response_ft, context_data)
            except Exception as e:
                print(f"⚠️ OpenRouter judge failed ({e}). Falling back to heuristic mock judge.")

        # 4. Fallback to heuristic mock judge
        return self.mock_judge.evaluate_pair(task, topic, user_prompt, response_base, response_ft, context_data)


def _parse_judge_json(raw_text: str) -> dict[str, Any]:
    """Cleanly parse JSON output from LLM judge."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex extraction
        base_s = re.search(r'"base_score":\s*([0-9.]+)', cleaned)
        ft_s = re.search(r'"ft_score":\s*([0-9.]+)', cleaned)
        win_s = re.search(r'"winner":\s*"([^"]+)"', cleaned)
        reas_s = re.search(r'"reasoning":\s*"([^"]+)"', cleaned)

        return {
            "base_score": float(base_s.group(1)) if base_s else 3.0,
            "ft_score": float(ft_s.group(1)) if ft_s else 3.0,
            "winner": win_s.group(1) if win_s else "tie",
            "reasoning": reas_s.group(1) if reas_s else "Parsed via fallback regex.",
        }

    return {
        "base_score": float(data.get("base_score", 3.0)),
        "ft_score": float(data.get("ft_score", 3.0)),
        "winner": str(data.get("winner", "tie")).lower(),
        "reasoning": str(data.get("reasoning", "")),
    }
