"""One SpaceXAI provider. Credentials stay on the server."""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from app.interpreter.prompts import SYSTEM_PROMPT, user_prompt
from app.interpreter.raw import Draft, LLMRaw, draft_from_llm
from app.llm.client import ChatClient, XAIChatClient
from app.llm.errors import ProviderBadResponse, ProviderUnavailable


class LLMProvider(Protocol):
    def interpret(self, text: str, language: str) -> Draft:
        """Return a structured draft. Callers still validate it."""


class XAIProvider:
    """SpaceXAI chat provider. The model id defaults to grok-4.7."""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-4.7",
        timeout_seconds: float = 30.0,
        client: ChatClient | None = None,
    ) -> None:
        self._client = client or XAIChatClient(
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
