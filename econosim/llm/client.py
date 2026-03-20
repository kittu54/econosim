"""OpenAI-compatible LLM client for EconoSim.

Supports any OpenAI-compatible API (OpenAI, Anthropic via proxy,
DeepSeek, Qwen, local models via Ollama/vLLM, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Create config from environment variables."""
        return cls(
            api_key=os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
        )


@dataclass
class LLMResponse:
    """Structured response from LLM."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        """Parse content as JSON, returning empty dict on failure."""
        try:
            # Handle markdown code blocks
            text = self.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = lines[1:]  # remove opening ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse LLM response as JSON")
            return {}


class LLMClient:
    """OpenAI-compatible LLM client with retry logic.

    Works with any provider that implements the OpenAI chat completions
    API format: OpenAI, Anthropic (via proxy), DeepSeek, Qwen,
    Ollama, vLLM, LiteLLM, etc.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of {"role": ..., "content": ...} messages.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            json_mode: Request JSON output format.
            model: Override default model.

        Returns:
            LLMResponse with content and metadata.
        """
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                return LLMResponse(
                    content=choice["message"]["content"],
                    model=data.get("model", ""),
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason", ""),
                    raw=data,
                )
            except (httpx.HTTPError, KeyError, IndexError) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(wait)

        raise ConnectionError(
            f"LLM request failed after {self.config.max_retries} retries: {last_error}"
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Simple completion with optional system prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, json_mode=json_mode)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls.

    Returns configurable responses or generates simple rule-based
    responses based on the prompt content.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default_response: str = '{"decision": "default", "reasoning": "mock response"}',
    ) -> None:
        self.config = LLMConfig(api_key="mock", base_url="http://mock")
        self._responses = list(responses) if responses else []
        self._default_response = default_response
        self._call_history: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Return mock response."""
        self._call_history.append({"messages": messages, "kwargs": kwargs})

        content = self._default_response
        if self._responses:
            content = self._responses.pop(0)

        return LLMResponse(
            content=content,
            model="mock",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            finish_reason="stop",
        )

    @property
    def call_count(self) -> int:
        return len(self._call_history)

    @property
    def last_call(self) -> dict[str, Any] | None:
        return self._call_history[-1] if self._call_history else None
