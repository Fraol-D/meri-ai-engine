"""Meri AI engine application. Interpretation only; no database and no Meri API client."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import Settings, load_settings
from app.interpreter.service import InterpreterService
from app.llm.provider import GeminiProvider, LLMProvider

_UNSET = object()


def build_provider(settings: Settings) -> LLMProvider | None:
    if not settings.gemini_api_key:
        return None
    return GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.state.interpreter = InterpreterService(resolved)
    app.include_router(router)
    return app


app = create_app()
