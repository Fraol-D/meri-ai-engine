"""Interpret request. business_id is not required and is ignored."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PreviousResult(BaseModel):
    """The clarification the caller is answering. Not stored by this service."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["clarification"] = "clarification"
    missing_fields: list[str] = Field(default_factory=list, max_length=12)
    question: str | None = Field(default=None, max_length=2000)


class InterpretContext(BaseModel):
    """Caller-supplied continuation. The engine does not keep a conversation."""

    model_config = ConfigDict(extra="ignore")

    original_text: str = Field(min_length=1, max_length=4000)
    previous_result: PreviousResult | None = None


class InterpretRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=4000)
    language: str = Field(
        default="en",
        pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{2,8})?$",
    )
    context: InterpretContext | None = None
