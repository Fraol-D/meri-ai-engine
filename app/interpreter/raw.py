"""Structured model output. Public responses are built from this, not returned raw."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

EventTypeName = Literal[
    "sale",
    "expense",
    "purchase",
    "inventory_adjustment",
    "customer_debt",
]
AmountScope = Literal["total", "unit", "ambiguous", "absent"]
DebtDirectionName = Literal["owed_to_business", "owed_by_business"]


class LLMRaw(BaseModel):
    """Schema sent to the provider. Unknown keys are dropped."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["create_event", "query", "clarification"]
    event_type: EventTypeName | None = None
    item: str | None = None
    quantity: int | float | None = None
    amount: int | float | None = None
    unit_price: int | float | None = None
    amount_scope: AmountScope | None = None
    currency: str | None = None
    customer: str | None = None
    supplier: str | None = None
    description: str | None = None
    category: str | None = None
    reason: str | None = None
    direction: DebtDirectionName | None = None
    date: str | None = None
    query: str | None = None
    vague_quantity: bool = False
    vague_amount: bool = False
    vague_item: bool = False

    @field_validator(
        "item",
        "currency",
        "customer",
        "supplier",
        "description",
        "category",
        "reason",
        "date",
        "query",
        mode="before",
    )
    @classmethod
    def blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("quantity", "amount", "unit_price", mode="before")
    @classmethod
    def reject_non_finite(cls, value: object) -> object:
        if value is None or isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, bool):
            raise ValueError("boolean is not a number")
        if isinstance(value, int | float) and value != value:
            raise ValueError("number is not finite")
        if isinstance(value, float) and value in {float("inf"), float("-inf")}:
            raise ValueError("number is not finite")
        return value


@dataclass
class Draft:
    kind: str
    confident: bool = False
    event_type: str | None = None
    item: str | None = None
    quantity: Decimal | None = None
    amount: Decimal | None = None
    unit_price: Decimal | None = None
    amount_scope: str | None = None
    currency: str | None = None
    customer: str | None = None
    supplier: str | None = None
    description: str | None = None
    category: str | None = None
    reason: str | None = None
    direction: str | None = None
    date: str | None = None
    query: str | None = None
    vague_quantity: bool = False
    vague_amount: bool = False
    vague_item: bool = False


def draft_from_llm(raw: LLMRaw) -> Draft:
    return Draft(
        kind=raw.kind,
        confident=False,
        event_type=raw.event_type,
        item=raw.item,
        quantity=_decimal(raw.quantity),
        amount=_decimal(raw.amount),
        unit_price=_decimal(raw.unit_price),
        amount_scope=raw.amount_scope,
        currency=raw.currency,
        customer=raw.customer,
        supplier=raw.supplier,
        description=raw.description,
        category=raw.category,
        reason=raw.reason,
        direction=raw.direction,
        date=raw.date,
        query=raw.query,
        vague_quantity=raw.vague_quantity,
        vague_amount=raw.vague_amount,
        vague_item=raw.vague_item,
    )


def _decimal(value: int | float | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, int):
        return Decimal(value)
    parsed = Decimal(str(value))
    if not parsed.is_finite():
        raise ValueError("number is not finite")
    return parsed
