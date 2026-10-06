"""Gemini chat client. The SDK is imported only when a request is sent."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from pydantic import ValidationError

from app.interpreter.raw import LLMRaw
from app.llm.errors import ProviderBadResponse, ProviderUnavailable

logger = logging.getLogger(__name__)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> LLMRaw | dict[str, object]:
        """Return structured JSON or a validated model."""


class GeminiChatClient:
    """Google Gemini structured-output client. The default model is gemini-3.1-flash-lite."""

    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is not set.")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def complete(self, messages: list[dict[str, str]]) -> LLMRaw:
        try:
            content = self._generate(messages)
        except ProviderBadResponse:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini request failed: %s", type(exc).__name__)
            raise ProviderUnavailable() from None
        return _parse_raw(content)

    def _generate(self, messages: list[dict[str, str]]) -> str:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini SDK import failed: %s", type(exc).__name__)
            raise ProviderUnavailable() from None

        system_text = next(item["content"] for item in messages if item["role"] == "system")
        user_text = next(item["content"] for item in messages if item["role"] == "user")
        timeout_ms = int(self._timeout * 1000)
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        response = client.models.generate_content(
            model=self._model,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_text,
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=LLMRaw.model_json_schema(),
            ),
        )
        content = response.text
        if not content or not content.strip():
            raise ProviderBadResponse()
        return content


def _parse_raw(content: str) -> LLMRaw:
    text = content.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    try:
        return LLMRaw.model_validate_json(text)
    except (ValidationError, ValueError):
        raise ProviderBadResponse() from None
