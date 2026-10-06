"""SpaceXAI chat client. The SDK is imported only when a request is sent."""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import ValidationError

from app.interpreter.raw import LLMRaw
from app.llm.errors import ProviderBadResponse, ProviderUnavailable

logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> LLMRaw | dict[str, object]:
        """Return structured JSON or a validated model."""


class XAIChatClient:
    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def complete(self, messages: list[dict[str, str]]) -> LLMRaw:
        try:
            from xai_sdk import Client
            from xai_sdk.chat import system, user
        except Exception as exc:  # noqa: BLE001
            logger.error("SpaceXAI SDK import failed: %s", type(exc).__name__)
            raise ProviderUnavailable() from None

        system_text = next(item["content"] for item in messages if item["role"] == "system")
        user_text = next(item["content"] for item in messages if item["role"] == "user")
        try:
            client = Client(api_key=self._api_key, timeout=self._timeout)
            chat = client.chat.create(
                model=self._model,
                response_format=LLMRaw,
                temperature=0,
                store_messages=False,
            )
            chat.append(system(system_text))
            chat.append(user(user_text))
            content = chat.sample().content
        except Exception as exc:  # noqa: BLE001
            logger.error("SpaceXAI request failed: %s", type(exc).__name__)
            raise ProviderUnavailable() from None

        try:
            return LLMRaw.model_validate_json(content)
        except (ValidationError, ValueError):
            raise ProviderBadResponse() from None
