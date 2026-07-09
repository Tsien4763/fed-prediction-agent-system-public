from __future__ import annotations

import json
import re
import time
from typing import Any

import requests


class TeacherClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: int = 90,
        allow_mock: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.allow_mock = allow_mock

    @property
    def is_mock(self) -> bool:
        return not self.api_key and self.allow_mock

    def chat_json(self, messages: list[dict[str, str]], schema_hint: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.is_mock:
            return self._mock_response(messages)
        content = self.chat(messages, response_format={"type": "json_object"} if schema_hint is not None else None)
        return extract_json_object(content)

    def chat(self, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None) -> str:
        if self.is_mock:
            return json.dumps(self._mock_response(messages), ensure_ascii=False)
        if not self.api_key:
            raise RuntimeError("DeepSeek API key is missing. Set DEEPSEEK_API_KEY or enable mock teacher.")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        body = self._post_chat_completion(
            f"{self.base_url}/v1/chat/completions",
            payload=payload,
        )
        return str(body["choices"][0]["message"]["content"])

    def _post_chat_completion(self, url: str, *, payload: dict[str, Any], max_attempts: int = 3) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable_status = status is None or status in {408, 429, 500, 502, 503, 504}
                if attempt >= max_attempts or not retryable_status:
                    break
                time.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"Teacher API request failed after {max_attempts} attempts") from last_error

    def _mock_response(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        text = "\n".join(message["content"] for message in messages).lower()
        inflation = 0.7 if "inflation" in text or "cpi" in text else 0.48
        labor = 0.35 if "labor" in text or "unemployment" in text else 0.25
        energy = 0.62 if "energy" in text or "oil" in text else 0.28
        hawkish = min(0.95, max(0.05, 0.25 + inflation * 0.45 + energy * 0.15 - labor * 0.15))
        return {
            "hawkish_dovish_score": round(hawkish * 2 - 1, 3),
            "inflation_concern": round(inflation, 3),
            "growth_outlook": round(0.35 - labor * 0.4, 3),
            "forward_guidance_strength": 0.32,
            "strategic_ambiguity": 0.48,
            "policy_stickiness": 0.72,
            "strategy": {
                "hawkish_signal_prob": round(hawkish, 3),
                "rate_hike_25bp_prob": round(max(0.05, hawkish - 0.42), 3),
                "hold_with_hawkish_statement_prob": round(min(0.9, hawkish + 0.15), 3),
                "remove_forward_guidance_prob": 0.67,
                "easing_signal_prob": round(max(0.02, 0.35 - hawkish), 3),
                "liquidity_support_prob": 0.18,
                "trade_or_sanction_pressure_prob": round(energy, 3),
            },
            "rationale": "Mock teacher response generated from keyword heuristics for local dry runs.",
            "evidence_chain": ["keyword heuristic: inflation/energy/labor"],
        }


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in teacher output: {text[:300]}")
    return json.loads(match.group(0))
