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
        if allow_mock:
            raise RuntimeError("Mock teacher mode is disabled; provide a real DeepSeek API key.")
        self.allow_mock = allow_mock

    @property
    def is_mock(self) -> bool:
        return False

    def chat_json(self, messages: list[dict[str, str]], schema_hint: dict[str, Any] | None = None) -> dict[str, Any]:
        content = self.chat(messages, response_format={"type": "json_object"} if schema_hint is not None else None)
        return extract_json_object(content)

    def chat(self, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("DeepSeek API key is missing. Set DEEPSEEK_API_KEY; mock fallback is disabled.")
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

def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in teacher output: {text[:300]}")
    return json.loads(match.group(0))
