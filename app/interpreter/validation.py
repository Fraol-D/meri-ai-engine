"""Semantic checks that run after rules or the model. The model is not the only gate."""

from __future__ import annotations

import re
from decimal import Decimal

from app.interpreter.normalization import (
    ArithmeticSafetyError,
    multiply_unit_price,
    normalize_currency,
    parse_iso_date,
    to_json_number,
)
from app.interpreter.raw import Draft
from app.interpreter.rules import (
    Evidence,
    analyze,
    draft_from_evidence,
    inventory_polarity,
    inventory_reason,
)
from app.schemas.responses import (
    ClarificationResult,
    CreateEventResult,
    EventData,
    InterpretationResult,
    QueryResult,
)

_ORDER = [
    "item",
    "description",
    "quantity",
    "amount",
    "amount_scope",
    "customer",
    "direction",
    "reason",
    "currency",
    "date",
]
_EVENTS = {"sale", "expense", "purchase", "inventory_adjustment", "customer_debt"}
_PHRASE_LIMIT = 80


def unknown_clarification() -> ClarificationResult:
    return ClarificationResult(
        type="clarification",
        question=(
            "Please state a sale, purchase, expense, inventory change, "
            "debt, or a business question."
        ),
        missing_fields=["intent"],
    )


def finalize(draft: Draft, text: str) -> InterpretationResult:
    """Apply source-text evidence, then accept an event or ask for what is missing."""
    evidence = analyze(text)
    if evidence.speech == "query" or (draft.kind == "query" and evidence.speech == "query"):
        return QueryResult(type="query", query=text.strip())
    if evidence.speech != "unknown":
        grounded = draft_from_evidence(evidence)
    else:
        grounded = ground_unrecognized(draft, evidence, text)
        if grounded.kind == "query":
            return QueryResult(type="query", query=text.strip())
    return decide(grounded, evidence)


def ground_unrecognized(draft: Draft, evidence: Evidence, text: str) -> Draft:
    """Drop model fields that the utterance does not support."""
    grounded = Draft(
        kind=draft.kind,
        confident=False,
        event_type=draft.event_type if draft.event_type in _EVENTS else None,
        item=_phrase(draft.item, text),
        description=_phrase(draft.description, text),
        customer=_phrase(draft.customer, text),
        supplier=_phrase(draft.supplier, text),
        category=_phrase(draft.category, text),
        reason=_phrase(draft.reason, text),
        direction=_direction(draft.direction, text),
        query=text.strip() if draft.kind == "query" else None,
    )
    if evidence.vague_item:
        grounded.item = None
        grounded.vague_item = True
    if evidence.vague_quantity:
        grounded.quantity = None
        grounded.vague_quantity = True
    elif evidence.quantity is not None:
        grounded.quantity = evidence.quantity
    if evidence.vague_amount:
        grounded.vague_amount = True
    elif evidence.amount_scope == "unit" and evidence.money is not None:
        grounded.unit_price = evidence.money
        grounded.amount_scope = "unit"
    elif evidence.amount_scope == "total" and evidence.money is not None:
        grounded.amount = evidence.money
        grounded.amount_scope = "total"
    elif evidence.amount_scope == "ambiguous":
        grounded.amount = None
        grounded.unit_price = None
        grounded.amount_scope = "ambiguous"
    else:
        # No unit or total cue. Keep a model amount only when this number is the
        # price the text named, and it is not also a quantity or a date piece.
        grounded.amount_scope = evidence.amount_scope or "absent"
        grounded.unit_price = None
        grounded.amount = _explicit_money_amount(draft, evidence)

    if evidence.currency is not None:
        grounded.currency = evidence.currency
    elif evidence.currency_surface is None:
        code = normalize_currency(draft.currency)
        grounded.currency = code if code and _currency_in_text(code, text) else None
    else:
        grounded.currency = None

    if evidence.date and not evidence.relative_date and not evidence.date_invalid:
        grounded.date = evidence.date
    else:
        grounded.date = parse_iso_date(draft.date) if draft.date and draft.date in text else None
        if evidence.relative_date or evidence.date_invalid:
            grounded.date = None

    if grounded.event_type is None:
        grounded.kind = "clarification"
    return grounded


def decide(draft: Draft, evidence: Evidence) -> InterpretationResult:
    _enforce_inventory_direction(draft, evidence)
    if draft.event_type not in _EVENTS:
        return unknown_clarification()

    _apply_unit_total(draft)
    missing = collect_missing(draft, evidence)
    if missing:
        return ClarificationResult(
            type="clarification",
            question=build_question(missing, draft, evidence),
            missing_fields=missing,
        )
    if draft.date is not None and parse_iso_date(draft.date) is None:
        return ClarificationResult(
            type="clarification",
            question="What date should I use? Please give a specific date as YYYY-MM-DD.",
            missing_fields=["date"],
        )
    return CreateEventResult(
        type="create_event",
        event_type=draft.event_type,  # type: ignore[arg-type]
        data=EventData(**_public_data(draft)),
    )


def collect_missing(draft: Draft, evidence: Evidence) -> list[str]:
    event = draft.event_type
    missing: list[str] = []
    if event in {"sale", "purchase", "inventory_adjustment"} and not draft.item:
        missing.append("item")
    if event == "inventory_adjustment" and (draft.quantity is None or draft.quantity == 0):
        missing.append("quantity")
    if event in {"sale", "purchase"} and (draft.quantity is None or draft.quantity <= 0):
        missing.append("quantity")

    if event in {"sale", "purchase"}:
        scope = draft.amount_scope or evidence.amount_scope
        explicit = scope in {"total", "unit"}
        if scope == "ambiguous" and "quantity" not in missing and draft.item:
            missing.append("amount_scope")
        elif (
            evidence.vague_quantity
            and evidence.money is not None
            and not explicit
            and "quantity" in missing
        ):
            missing.append("amount_scope")
        if "amount_scope" not in missing and not _amount_is_known(draft, scope, missing):
            missing.append("amount")

    if event == "expense":
        if not draft.description:
            missing.append("description")
        if draft.amount is None or draft.amount <= 0:
            missing.append("amount")
    if event == "customer_debt":
        if not draft.customer:
            missing.append("customer")
        if draft.amount is None or draft.amount <= 0:
            missing.append("amount")
        if draft.direction not in {"owed_to_business", "owed_by_business"}:
            missing.append("direction")
    if event == "inventory_adjustment" and not draft.reason:
        missing.append("reason")
    if evidence.relative_date or evidence.date_invalid:
        missing.append("date")
    if evidence.currency_unrecognized:
        missing.append("currency")
    return [name for name in _ORDER if name in missing]


def build_question(missing: list[str], draft: Draft, evidence: Evidence) -> str:
    event = draft.event_type or evidence.speech
    item = draft.item or "items"
    names = set(missing)

    if "amount_scope" in names and "quantity" not in names and "item" not in names:
        question = _scope_question(evidence, item)
        extras = [name for name in missing if name != "amount_scope"]
        if not extras:
            return question
        return f"{question} {_join_parts(extras, draft, evidence, item)}"

    if names == {"item", "quantity", "amount"} and event == "sale":
        return "What did you sell, how many, and for how much?"
    if names == {"item", "quantity", "amount"} and event == "purchase":
        return "What did you buy, how many, and for how much?"
    if names == {"quantity", "amount"} and event == "sale":
        return f"How many {item} did you sell, and for how much?"
    if names == {"quantity", "amount"} and event == "purchase":
        return f"How many {item} did you buy, and for how much?"
    if names == {"description", "amount"}:
        return "What did you spend, and how much was it?"
    if names == {"quantity", "amount_scope"}:
        price = _money_phrase(evidence)
        verb = "buy" if event == "purchase" else "sell"
        return (
            f"How many {item} did you {verb}? "
            f"Say whether {price} is the total or the price per {_singular(item)}."
        )
    if len(missing) == 1:
        return _one_question(missing[0], draft, evidence, item)
    return _join_parts(missing, draft, evidence, item)


def _apply_unit_total(draft: Draft) -> None:
    scope = draft.amount_scope
    if draft.event_type not in {"sale", "purchase"} or scope != "unit":
        return
    if draft.quantity is None or draft.quantity <= 0 or draft.unit_price is None:
        return
    try:
        draft.amount = multiply_unit_price(draft.quantity, draft.unit_price)
    except ArithmeticSafetyError:
        draft.amount = None


def _amount_is_known(draft: Draft, scope: str, missing: list[str]) -> bool:
    unit_known = draft.unit_price is not None and draft.unit_price > 0
    if scope == "unit" and "quantity" in missing and unit_known:
        return True
    return draft.amount is not None and draft.amount > 0


def _public_data(draft: Draft) -> dict[str, object]:
    event = draft.event_type
    data: dict[str, object] = {}
    if event in {"sale", "purchase", "inventory_adjustment"}:
        data["item"] = draft.item
        data["quantity"] = to_json_number(draft.quantity) if draft.quantity is not None else None
    if event in {"sale", "purchase", "expense", "customer_debt"}:
        data["amount"] = to_json_number(draft.amount) if draft.amount is not None else None
    if event == "expense":
        data["description"] = draft.description
        if draft.category:
            data["category"] = draft.category
    if event == "inventory_adjustment":
        data["reason"] = draft.reason
    if event == "customer_debt":
        data["customer"] = draft.customer
        data["direction"] = draft.direction
    if event == "sale" and draft.customer:
        data["customer"] = draft.customer
    if event == "purchase" and draft.supplier:
        data["supplier"] = draft.supplier
    if draft.currency and event != "inventory_adjustment":
        data["currency"] = draft.currency
    if draft.date:
        data["date"] = draft.date
    return {key: value for key, value in data.items() if value is not None}


def _scope_question(evidence: Evidence, item: str) -> str:
    money = _money_phrase(evidence)
    quantity = evidence.quantity_surface or (
        to_json_number(evidence.quantity) if evidence.quantity is not None else "them"
    )
    return (
        f"Is {money} the total for all {quantity} {item}, "
        f"or {money} per {_singular(item)}?"
    )


def _money_phrase(evidence: Evidence) -> str:
    money = evidence.money_surface
    if money is None and evidence.money is not None:
        money = str(to_json_number(evidence.money))
    if money is None:
        money = "that price"
    if evidence.currency_surface:
        return f"{money} {evidence.currency_surface}"
    return str(money)


def _one_question(name: str, draft: Draft, evidence: Evidence, item: str) -> str:
    event = draft.event_type or evidence.speech
    if name == "item":
        if event == "purchase":
            return "What did you buy?"
        if event == "inventory_adjustment":
            return "Which item changed?"
        return "What did you sell?"
    if name == "description":
        return "What was the expense for?"
    if name == "quantity":
        if event == "purchase":
            return f"How many {item} did you buy?"
        if event == "inventory_adjustment":
            reason = draft.reason or "changed"
            return f"How many {item} were {reason}?"
        return f"How many {item} did you sell?"
    if name == "amount":
        if event == "purchase":
            return "How much was the purchase?"
        if event == "expense":
            return "How much was the expense?"
        if event == "customer_debt":
            return _debt_amount_question(draft)
        return "How much was the sale?"
    if name == "customer":
        return "Who is the customer?"
    if name == "direction":
        if draft.customer:
            return f"Does {draft.customer} owe you, or do you owe {draft.customer}?"
        return "Who owes whom?"
    if name == "reason":
        return "Why did the inventory change?"
    if name == "currency":
        return "Which currency was that?"
    if name == "date":
        return "What date should I use? Please give a specific date as YYYY-MM-DD."
    if name == "amount_scope":
        return _scope_question(evidence, item)
    return "Please state the missing detail."


def _join_parts(names: list[str], draft: Draft, evidence: Evidence, item: str) -> str:
    parts = [_one_question(name, draft, evidence, item) for name in names]
    return " ".join(parts)


def _debt_amount_question(draft: Draft) -> str:
    name = draft.customer or "the customer"
    if draft.direction == "owed_to_business":
        return f"How much does {name} owe you?"
    if draft.direction == "owed_by_business":
        return f"How much do you owe {name}?"
    return "How much is the debt?"


def _singular(noun: str) -> str:
    if noun.endswith("sses"):
        return noun
    if len(noun) > 2 and noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def _phrase(value: str | None, text: str) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n.,!?:;\"'")
    if not cleaned or len(cleaned) > _PHRASE_LIMIT:
        return None
    if cleaned.lower() not in text.lower():
        return None
    return cleaned


def _direction(value: str | None, text: str) -> str | None:
    if re.search(r"\bowes\s+me\b|\bowed\s+me\b", text, re.IGNORECASE):
        return "owed_to_business"
    if re.search(r"\bi\s+owe\b", text, re.IGNORECASE):
        return "owed_by_business"
    if value in {"owed_to_business", "owed_by_business"} and re.search(r"\bowes?\b", text, re.I):
        return None
    return None


def _enforce_inventory_direction(draft: Draft, evidence: Evidence) -> None:
    """The utterance, not the model, decides whether stock went up or down."""
    if draft.event_type != "inventory_adjustment":
        return
    polarity = inventory_polarity(evidence.original)
    reason = inventory_reason(evidence.original)
    if polarity == "loss":
        if draft.quantity is not None:
            draft.quantity = -abs(draft.quantity)
        if reason is not None:
            draft.reason = reason
        return
    if polarity == "gain":
        if draft.quantity is not None:
            draft.quantity = abs(draft.quantity)
        if reason is not None:
            draft.reason = reason
        return
    draft.quantity = None
    draft.reason = None
    if polarity != "ambiguous":
        draft.event_type = None


def _explicit_money_amount(draft: Draft, evidence: Evidence) -> Decimal | None:
    """Accept an amount only when the text calls that number a price.

    A quantity, a date piece, or any other number that merely occurs in the
    utterance is not an amount. Unit-price totals are applied later from the
    parsed quantity and the explicit unit price.
    """
    if evidence.vague_amount or evidence.money is None or draft.amount is None:
        return None
    if evidence.quantity is not None:
        return None
    if draft.amount != evidence.money:
        return None
    if any(draft.amount == number for number in evidence.quantity_values):
        return None
    return evidence.money


def _currency_in_text(code: str, text: str) -> bool:
    if code == "ETB":
        return re.search(r"\b(birr|etb|ethiopian\s+birr)\b", text, re.IGNORECASE) is not None
    return code.lower() in text.lower()
