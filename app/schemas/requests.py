"""Interpret request. business_id is not required and is ignored."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InterpretRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=4000)
    language: str = Field(
        default="en",
        pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{2,8})?$",
    )
