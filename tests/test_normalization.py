"""Currency, dates, debt direction, and unit-price arithmetic."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.interpreter.normalization import (
    ArithmeticSafetyError,
    multiply_unit_price,
    normalize_currency,
    to_json_number,
)
from app.interpreter.service import interpret_text


def _dump(text: str) -> dict:
    return interpret_text(text, "en").model_dump(mode="json", exclude_none=True)


def test_currency_codes() -> None:
    assert normalize_currency("birr") == "ETB"
    assert normalize_currency("ETB") == "ETB"
    assert normalize_currency("Ethiopian birr") == "ETB"
    assert normalize_currency("USD") == "USD"
    assert normalize_currency("dollars") == "USD"
    assert normalize_currency(None) is None


def test_birr_is_never_returned_as_birr() -> None:
    payload = _dump("I sold five shirts for 900 Ethiopian birr total.")
    assert payload["data"]["currency"] == "ETB"
    assert "BIRR" not in str(payload).upper() or payload["data"]["currency"] == "ETB"
    assert payload["data"]["currency"] != "BIRR"


def test_missing_currency_is_omitted() -> None:
    payload = _dump("I sold five shirts for 900 total.")
    assert "currency" not in payload["data"]
    assert payload["data"]["amount"] == 900


def test_other_currency_is_preserved() -> None:
    payload = _dump("I sold five shirts for 10 USD total.")
    assert payload["data"]["currency"] == "USD"
    assert payload["data"]["amount"] == 10


def test_unit_price_multiplication_uses_exact_decimals() -> None:
    assert multiply_unit_price(Decimal(5), Decimal(900)) == Decimal(4500)
    assert multiply_unit_price(Decimal(20), Decimal(250)) == Decimal(5000)
    assert to_json_number(Decimal("4500")) == 4500
    assert isinstance(to_json_number(Decimal("4500")), int)


def test_unsafe_multiplication_is_rejected() -> None:
    with pytest.raises(ArithmeticSafetyError):
        multiply_unit_price(Decimal(0), Decimal(10))
    with pytest.raises(ArithmeticSafetyError):
        multiply_unit_price(Decimal(-2), Decimal(10))


def test_concrete_date_is_preserved() -> None:
    payload = _dump("I sold five shirts for 900 birr total on 2026-01-15.")
    assert payload["data"]["date"] == "2026-01-15"
    assert payload["data"]["amount"] == 900


def test_month_name_date_is_normalized() -> None:
    payload = _dump("I sold five shirts for 900 birr total on January 5, 2026.")
    assert payload["data"]["date"] == "2026-01-05"


def test_relative_date_is_not_turned_into_today() -> None:
    payload = _dump("I sold five shirts for 900 birr total yesterday.")
    assert payload["type"] == "clarification"
    assert "date" in payload["missing_fields"]
    assert date.today().isoformat() not in payload["question"]
    assert "data" not in payload


def test_unclear_debt_direction_is_not_guessed() -> None:
    payload = _dump("John owes 500 birr.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["direction"]
    assert "owed_" not in payload["question"]


def test_explicit_total_without_quantity_does_not_invent_one() -> None:
    payload = _dump("I sold shirts for 900 birr total.")
    assert payload["missing_fields"] == ["quantity"]
    assert "data" not in payload


def test_explicit_unit_price_without_quantity_does_not_invent_one() -> None:
    payload = _dump("I sold shirts for 900 birr each.")
    assert payload["missing_fields"] == ["quantity"]
    assert "4500" not in str(payload)
    assert "data" not in payload


def test_lost_some_shirts_has_no_quantity() -> None:
    payload = _dump("I lost some shirts.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["quantity"]
    assert not any(character.isdigit() for character in payload["question"])
