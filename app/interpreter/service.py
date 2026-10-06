"""Orchestrates rules, the provider, and semantic validation."""

from __future__ import annotations

from app.interpreter.raw import Draft
from app.interpreter.rules import analyze, draft_from_evidence
from app.interpreter.validation import finalize, unknown_clarification
from app.llm.provider import LLMProvider
from app.schemas.responses import InterpretationResult


class EmptyUtterance(ValueError):
    """The request text was empty after trimming."""


class InterpreterService:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    def interpret(self, text: str, language: str = "en") -> InterpretationResult:
        return interpret_text(text, language, self.provider)


def interpret_text(
    text: str,
    language: str = "en",
    provider: LLMProvider | None = None,
) -> InterpretationResult:
    cleaned = text.strip()
    if not cleaned:
        raise EmptyUtterance()
    evidence = analyze(cleaned)
    draft: Draft
    if evidence.speech == "unknown":
        if provider is None:
            return unknown_clarification()
        draft = provider.interpret(cleaned, language)
    else:
        draft = draft_from_evidence(evidence)
    return finalize(draft, cleaned)
