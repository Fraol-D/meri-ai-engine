"""One Gemini provider. Credentials stay on the server."""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from app.interpreter.prompts import SYSTEM_PROMPT, user_prompt
from app.interpreter.raw import Draft, LLMRaw, draft_from_llm
from app.llm.client import ChatClient, GeminiChatClient
from app.llm.errors import ProviderBadResponse, ProviderUnavailable

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


class LLMProvider(Protocol):
    def interpret(self, text: str, language: str) -> Draft:
        """Return a structured draft. Callers still validate it."""


class GeminiProvider:
    """Gemini chat provider. The model id defaults to gemini-3.1-flash-lite."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout_seconds: float = 30.0,
        client: ChatClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GEMINI_API_KEY is not set.")
        self._client = client or GeminiChatClient(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )

    def interpret(self, text: str, language: str) -> Draft:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(text, language)},
        ]
        for _ in range(2):
            try:
                payload = self._client.complete(messages)
            except ProviderUnavailable:
                raise
            except ProviderBadResponse:
                continue
            try:
                raw = payload if isinstance(payload, LLMRaw) else LLMRaw.model_validate(payload)
                return draft_from_llm(raw)
            except (ValidationError, ValueError):
                continue
        raise ProviderBadResponse()
