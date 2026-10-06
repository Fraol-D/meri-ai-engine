"""Public interpretation results. No model-only fields are accepted."""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EventType = Literal[
    "sale",
    "expense",
    "purchase",
    "inventory_adjustment",
    "customer_debt",
]
DebtDirection = Literal["owed_to_business", "owed_by_business"]

_REQUIRED: dict[str, set[str]] = {
    "sale": {"item", "quantity", "amount"},
    "expense": {"description", "amount"},
    "purchase": {"item", "quantity", "amount"},
    "inventory_adjustment": {"item", "quantity", "reason"},
    "customer_debt": {"customer", "amount", "direction"},
}
_ALLOWED: dict[str, set[str]] = {
    "sale": {"item", "quantity", "amount", "currency", "customer", "date"},
    "expense": {"description", "amount", "currency", "category", "date"},
    "purchase": {"item", "quantity", "amount", "currency", "supplier", "date"},
    "inventory_adjustment": {"item", "quantity", "reason", "date"},
    "customer_debt": {"customer", "amount", "direction", "currency", "date"},
}
_INTERNAL_TERMS = (
    "amount_scope",
    "owed_to_business",
    "owed_by_business",
    "create_event",
    "event_type",
    "vague_",
)


class EventData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str | None = None
    quantity: int | float | None = None
    amount: int | float | None = None
    currency: str | None = None
    customer: str | None = None
    supplier: str | None = None
    description: str | None = None
    category: str | None = None
    reason: str | None = None
    direction: DebtDirection | None = None
    date: str | None = None


class CreateEventResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["create_event"]
    event_type: EventType
    data: EventData

    @model_validator(mode="after")
    def data_matches_contract(self) -> CreateEventResult:
        present = {key for key, value in self.data.model_dump().items() if value is not None}
        required = _REQUIRED[self.event_type]
        allowed = _ALLOWED[self.event_type]
        if not required <= present or not present <= allowed:
            raise ValueError("event data does not match the backend contract")
        _check_ranges(self.event_type, self.data)
        if self.data.currency is not None and (
            self.data.currency == "BIRR" or re.fullmatch(r"[A-Z]{3}", self.data.currency) is None
        ):
            raise ValueError("currency must be an ISO code")
        if self.data.date is not None:
            date.fromisoformat(self.data.date)
        return self


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["query"]
    query: str = Field(min_length=1)


class ClarificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["clarification"]
    question: str = Field(min_length=1)
    missing_fields: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def question_is_user_facing(self) -> ClarificationResult:
        lowered = self.question.lower()
        for term in _INTERNAL_TERMS:
            if term in lowered:
                raise ValueError("clarification exposed an internal term")
        return self


InterpretationResult = Annotated[
    CreateEventResult | QueryResult | ClarificationResult,
    Field(discriminator="type"),
]


def _check_ranges(event_type: str, data: EventData) -> None:
    if event_type in {"sale", "purchase"}:
        if data.quantity is None or data.quantity <= 0 or data.amount is None or data.amount <= 0:
            raise ValueError("sale and purchase need a positive quantity and amount")
    elif event_type == "expense":
        if data.amount is None or data.amount <= 0:
            raise ValueError("expense amount must be positive")
    elif event_type == "inventory_adjustment":
        if data.quantity is None or data.quantity == 0:
            raise ValueError("inventory quantity must not be zero")
    elif event_type == "customer_debt":
        if data.amount is None or data.amount <= 0:
            raise ValueError("debt amount must be positive")
        if data.direction not in {"owed_to_business", "owed_by_business"}:
            raise ValueError("debt direction is not allowed")
