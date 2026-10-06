"""Meri AI engine application. Interpretation only; no database and no Meri API client."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, load_settings
from app.interpreter.service import InterpreterService
from app.llm.provider import LLMProvider, XAIProvider

_UNSET = object()


def build_provider(settings: Settings) -> LLMProvider | None:
    if not settings.xai_api_key:
        return None
    return XAIProvider(
        api_key=settings.xai_api_key,
        model=settings.xai_model,
        timeout_seconds=settings.xai_timeout_seconds,
    )


def create_app(provider: LLMProvider | None | object = _UNSET) -> FastAPI:
    settings = load_settings()
    resolved: LLMProvider | None
    if provider is _UNSET:
        resolved = build_provider(settings)
    else:
        resolved = provider  # type: ignore[assignment]
    app = FastAPI(
        title="Meri AI Engine",
        version="0.1.0",
        summary="Interpret business speech into events, queries, or clarifications.",
    )
    app.state.interpreter = InterpreterService(resolved)
    app.include_router(router)
    return app


app = create_app()
