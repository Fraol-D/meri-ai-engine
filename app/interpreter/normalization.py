"""Deterministic currency, date, and unit-price normalization."""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

MAX_AMOUNT = Decimal("1000000000000")
MAX_QUANTITY = Decimal("1000000000")

_CURRENCY = {
    "birr": "ETB",
    "etb": "ETB",
    "ethiopian birr": "ETB",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
}

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
MONTH_FIRST_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
DAY_FIRST_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r",?\s+(\d{4})\b",
    re.IGNORECASE,
)


class ArithmeticSafetyError(ValueError):
    """Unit-price multiplication was not a safe total."""


def normalize_currency(surface: str | None) -> str | None:
    """Map a stated currency to an ISO code. Unknown text returns None."""
    if surface is None:
        return None
    key = re.sub(r"\s+", " ", surface).strip().lower()
    if not key:
        return None
    if key in _CURRENCY:
        return _CURRENCY[key]
    if re.fullmatch(r"[a-z]{3}", key):
        return key.upper()
    return None


def currency_display(surface: str, code: str) -> str:
    if code == "ETB" and surface.lower() != "etb":
        return re.sub(r"\s+", " ", surface).strip().lower()
    if len(surface) == 3 and surface.isalpha():
        return surface.upper()
    return surface


def multiply_unit_price(quantity: Decimal, unit_price: Decimal) -> Decimal:
    """Convert an explicit unit price into one transaction total."""
    if quantity <= 0 or unit_price <= 0:
        raise ArithmeticSafetyError("quantity and unit price must be positive")
    if abs(quantity) > MAX_QUANTITY or unit_price > MAX_AMOUNT:
        raise ArithmeticSafetyError("value is too large")
    try:
        product = quantity * unit_price
    except InvalidOperation as exc:
        raise ArithmeticSafetyError("multiplication failed") from exc
    if not product.is_finite() or product <= 0 or product > MAX_AMOUNT:
        raise ArithmeticSafetyError("total is not usable")
    exponent = product.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -2:
        product = product.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return product


def to_json_number(value: Decimal) -> int | float:
    integral = value.to_integral_value(rounding=ROUND_HALF_UP)
    if value == integral:
        return int(integral)
    return float(value)


def parse_iso_date(value: str) -> str | None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    if parsed.isoformat() != value:
        return None
    return value


def _build_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def find_concrete_dates(text: str) -> tuple[str | None, bool, list[tuple[int, int]]]:
    """Return one ISO date, whether the date text was invalid, and spans to mask.

    Relative phrases are not converted here.
    """
    spans: list[tuple[int, int]] = []
    found: list[str] = []
    invalid = False

    for match in ISO_DATE_RE.finditer(text):
        spans.append(match.span())
        parsed = parse_iso_date(match.group(1))
        if parsed is None:
            invalid = True
        else:
            found.append(parsed)

    for match in MONTH_FIRST_RE.finditer(text):
        spans.append(match.span())
        parsed = _build_date(
            int(match.group(3)),
            _MONTHS[match.group(1).lower()],
            int(match.group(2)),
        )
        if parsed is None:
            invalid = True
        else:
            found.append(parsed)

    for match in DAY_FIRST_RE.finditer(text):
        spans.append(match.span())
        parsed = _build_date(
            int(match.group(3)),
            _MONTHS[match.group(2).lower()],
            int(match.group(1)),
        )
        if parsed is None:
            invalid = True
        else:
            found.append(parsed)

    unique = list(dict.fromkeys(found))
    if len(unique) > 1:
        return None, True, spans
    if invalid and not unique:
        return None, True, spans
    if len(unique) == 1:
        return unique[0], False, spans
    return None, False, spans


def mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            chars[index] = " "
    return "".join(chars)
