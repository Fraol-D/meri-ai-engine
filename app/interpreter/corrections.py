"""Apply an explicit correction without treating every later number as one.

The service sees one accumulated utterance. The first quantity and the first
price stay in place unless a later clause states a replacement. A rejection
with no replacement does not keep the rejected event.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from app.interpreter.normalization import (
    DAY_FIRST_RE,
    ISO_DATE_RE,
    MONTH_FIRST_RE,
    find_concrete_dates,
    normalize_currency,
)
from app.interpreter.rules import (
    CURRENCY_RE,
    NUMBER_RE,
    RELATIVE_DATE_RE,
    TOTAL_RE,
    UNIT_CUE_RE,
    _OMITTED_DATE_RE,
    Evidence,
    _is_money,
    _parse_number,
    analyze,
)
from app.schemas.responses import ClarificationResult

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_CURRENCY = (
    r"ethiopian\s+birr|birr|etb|us\s+dollars?|dollars?|usd|euros?|eur|pounds?|gbp"
)
_SCOPE = r"each(?:\s+one)?|apiece|a\s+piece|per\s+\w+|in\s+total|altogether|total"
_STRONG_REJECTION = re.compile(
    r"^(?:no+\s*[,:]|no+\s+(?=i\b|the\b|that\b|this\b|it\b|they\b|those\b|"
    r"quantity\b|qty\b|price\b|amount\b|item\b|date\b)|nope\b|wrong\b)",
    re.IGNORECASE,
)
_PREFACE = re.compile(
    r"^(?:actually|correction|instead|i\s+meant|i\s+mean)\b",
    re.IGNORECASE,
)
_CUE = re.compile(
    r"^(?:"
    r"no+\s*[,:]|no+\s+(?=i\b|the\b|that\b|this\b|it\b|they\b|those\b|"
    r"quantity\b|qty\b|price\b|amount\b|item\b|date\b)|"
    r"nope\b[\s,:-]*|wrong\b[\s,:-]*|"
    r"actually\b[\s,:-]*|correction\b[\s,:-]*|instead\b[\s,:-]*|"
    r"i\s+meant\b[\s,:-]*|i\s+mean\b[\s,:-]*"
    r")\s*",
    re.IGNORECASE,
)
_FIELD_START = re.compile(
    r"^(?:the\s+)?(?:quantity|qty|unit\s+price|price|amount|total|item|date)\s+"
    r"(?:is|was|should\s+be|to)\b",
    re.IGNORECASE,
)
_EMPTY_REJECTION = re.compile(
    r"^(?:(?:that|this|it)(?:'s|\s+is)?\s+)?"
    r"(?:wrong|incorrect|not\s+right|a\s+mistake)\s*$",
    re.IGNORECASE,
)
_QUANTITY_IS = re.compile(
    rf"\b(?:the\s+)?(?:quantity|qty)\s+(?:is|was|should\s+be|to)\s+"
    rf"(?P<number>{_NUMBER})\b",
    re.IGNORECASE,
)
_PRICE_IS = re.compile(
    rf"\b(?:the\s+)?(?:unit\s+price|price)\s+(?:is|was|should\s+be|to)\s+"
    rf"(?P<number>{_NUMBER})(?:\s+(?P<currency>{_CURRENCY}))?"
    rf"(?:\s+(?P<scope>{_SCOPE}))?",
    re.IGNORECASE,
)
_AMOUNT_IS = re.compile(
    rf"\b(?:the\s+)?(?:amount|total)\s+(?:is|was|should\s+be|to)\s+"
    rf"(?P<number>{_NUMBER})(?:\s+(?P<currency>{_CURRENCY}))?"
    rf"(?:\s+(?:total|in\s+total|altogether))?",
    re.IGNORECASE,
)
_THEY_WERE_PRICE = re.compile(
    rf"^(?:they|it|those|these)\s+(?:were|are|was|cost|costs)\s+"
    rf"(?P<number>{_NUMBER})(?:\s+(?P<currency>{_CURRENCY}))?"
    rf"(?:\s+(?P<scope>{_SCOPE}))?\s*$",
    re.IGNORECASE,
)
_ITEM_IS = re.compile(
    r"\b(?:the\s+)?item\s+(?:is|was|should\s+be)\s+(?P<item>[A-Za-z][\w'-]*)",
    re.IGNORECASE,
)
_DATE_IS = re.compile(
    r"\b(?:the\s+)?date\s+(?:is|was|should\s+be)\s+(?P<rest>.+)$",
    re.IGNORECASE,
)
_AND_PRICE = re.compile(
    r",?\s+and\s+(?:they|it|those|these)\s+(?:were|are|was|cost|costs)\s+"
    r"(?P<price>.+)$",
    re.IGNORECASE,
)
_TRADE_VERB = re.compile(
    r"\b(?:sold|sell|selling|sale|bought|buy|buying|purchased|purchase|"
    r"spent|spend|spending)\b",
    re.IGNORECASE,
)
# A comma or semicolon can introduce a correction without a new sentence.
# "3,000" has no space, so it does not match. "no problem" is not a cue.
# Do not split inside stacked cues like "No, I meant 20." or "No, actually ...".
_CLAUSE_BREAK = re.compile(
    r"(?:(?<=[.!?])\s+"
    r"|(?<!\bno[,;])(?<!\bnope[,;])(?<!\bwrong[,;])"
    r"(?<!\bactually[,;])(?<!\bcorrection[,;])(?<!\binstead[,;])"
    r"(?<=[,;])\s+(?=(?:nope|no|wrong|actually|correction|instead|"
    r"i\s+meant|i\s+mean)\b))",
    re.IGNORECASE,
)
_NUMERIC_ITEM = re.compile(r"\d[\d,]*(?:\.\d+)?\Z")
_AMBIGUOUS_NUMBER_CORRECTION = re.compile(
    rf"^(?:(?:make|change)\s+(?:it|that)\s+(?:to\s+)?|(?:it\s+(?:is|was|should\s+be)\s+|to\s+|be\s+)?)?"
    rf"(?P<number>{NUMBER_RE.pattern})"
    rf"(?:\s+(?P<currency>{_CURRENCY}))?\s*$",
    re.IGNORECASE,
)


@dataclass
class CorrectionRevision:
    """Utterance after correction clauses, or a rejection with nothing to use."""

    text: str
    unresolved: bool = False
    ambiguous_number: str | None = None


@dataclass
class _Change:
    quantity: Decimal | None = None
    money: Decimal | None = None
    scope: str | None = None
    item: str | None = None
    currency_surface: str | None = None
    date: str | None = None
    relative_phrase: str | None = None
    clear_date: bool = False
    utterance: str | None = None

    def any(self) -> bool:
        return any(
            (
                self.quantity is not None,
                self.money is not None,
                self.item,
                self.date,
                self.relative_phrase,
                self.clear_date,
                self.utterance,
            )
        )

    def merge(self, newer: _Change) -> _Change:
        """A later replacement wins for each field it states."""
        date = self.date
        relative = self.relative_phrase
        clear_date = self.clear_date
        if newer.date is not None:
            date = newer.date
            relative = None
            clear_date = False
        elif newer.relative_phrase is not None:
            date = None
            relative = newer.relative_phrase
            clear_date = False
        elif newer.clear_date:
            date = None
            relative = None
            clear_date = True
        return _Change(
            quantity=newer.quantity if newer.quantity is not None else self.quantity,
            money=newer.money if newer.money is not None else self.money,
            scope=newer.scope if newer.scope is not None else self.scope,
            item=newer.item if newer.item is not None else self.item,
            currency_surface=(
                newer.currency_surface
                if newer.currency_surface is not None
                else self.currency_surface
            ),
            date=date,
            relative_phrase=relative,
            clear_date=clear_date,
            utterance=newer.utterance if newer.utterance is not None else self.utterance,
        )


def partition_corrections(text: str) -> tuple[str, list[str]]:
    """Split ordinary sentences from explicit correction sentences."""
    return _split_clauses(text)


def revise_corrections(text: str) -> CorrectionRevision:
    """Rewrite text when it contains an explicit correction clause."""
    base, clauses = _split_clauses(text)
    if not clauses:
        return CorrectionRevision(text)
    # "The total is 900 birr" on its own is not a correction of an earlier sale.
    if (
        not base.strip()
        and not any(_TRADE_VERB.search(_strip_cues(clause)) for clause in clauses)
        and not any(_ambiguous_number_value(clause) is not None for clause in clauses)
    ):
        return CorrectionRevision(text)
    rewritten, ambiguous_number = _apply_clauses(base, clauses)
    if rewritten is None:
        return CorrectionRevision(
            base,
            unresolved=True,
            ambiguous_number=ambiguous_number,
        )
    return CorrectionRevision(rewritten)


def replacement_clarification(
    text: str,
    ambiguous_number: str | None = None,
) -> ClarificationResult:
    """Ask for a replacement. Do not return the rejected transaction."""
    evidence = analyze(text) if text.strip() else Evidence(original=text)
    if ambiguous_number and evidence.speech in {"sale", "purchase"}:
        return ClarificationResult(
            type="clarification",
            question=(
                f"Does {ambiguous_number} refer to the quantity, "
                "the unit price, or the total amount?"
            ),
            missing_fields=["quantity", "amount"],
        )
    if evidence.speech in {"sale", "purchase"}:
        return ClarificationResult(
            type="clarification",
            question="What is the correct item, quantity, and price?",
            missing_fields=["item", "quantity", "amount"],
        )
    if evidence.speech == "expense":
        return ClarificationResult(
            type="clarification",
            question="What is the correct expense and amount?",
            missing_fields=["description", "amount"],
        )
    if ambiguous_number:
        return ClarificationResult(
            type="clarification",
            question=(
                f"Does {ambiguous_number} refer to the quantity, "
                "the unit price, or the total amount?"
            ),
            missing_fields=["quantity", "amount"],
        )
    return ClarificationResult(
        type="clarification",
        question="What should I record instead?",
        missing_fields=["intent"],
    )


def _split_clauses(text: str) -> tuple[str, list[str]]:
    sentences = _sentences(text)
    ordinary: list[str] = []
    corrections: list[str] = []
    for sentence in sentences:
        if _is_correction_sentence(sentence):
            corrections.append(sentence)
        else:
            ordinary.append(sentence)
    if not corrections:
        return text, []
    return " ".join(ordinary).strip(), corrections


def _sentences(text: str) -> list[str]:
    parts = _CLAUSE_BREAK.split(text.strip())
    sentences: list[str] = []
    for part in parts:
        piece = part.strip(" \t\r\n,;")
        if piece:
            sentences.append(piece)
    return sentences


def _is_correction_sentence(sentence: str) -> bool:
    if _STRONG_REJECTION.match(sentence) or _FIELD_START.match(sentence):
        return True
    if _PREFACE.match(sentence) is None:
        return False
    # A preface is a correction only when a replacement or an ambiguous
    # bare-number correction ("Actually, 20", "I meant 20") can be read from it.
    # "Actually, the weather" must not throw away the transaction.
    return (
        _parse_clause(sentence) is not None
        or _ambiguous_number_value(sentence) is not None
    )


def _apply_clauses(base: str, clauses: list[str]) -> tuple[str | None, str | None]:
    change = _Change()
    rejected = False
    ambiguous_number: str | None = None
    for clause in clauses:
        parsed = _parse_clause(clause)
        strong = _STRONG_REJECTION.match(clause) is not None
        if parsed is None:
            bare_num = _ambiguous_number_value(clause)
            if bare_num is not None:
                rejected = True
                ambiguous_number = bare_num
                change = _Change()
            elif strong or _is_empty_rejection(_strip_cues(clause)):
                rejected = True
                ambiguous_number = None
                change = _Change()
            continue
        change = change.merge(parsed)
        rejected = False
        ambiguous_number = None
    if rejected or not change.any():
        return (None, ambiguous_number) if rejected else (base, None)
    return _rewrite(base, change), None


def _ambiguous_number_value(sentence: str) -> str | None:
    if not (_STRONG_REJECTION.match(sentence) or _PREFACE.match(sentence)):
        return None
    piece = _strip_cues(sentence)
    if not piece or _is_empty_rejection(piece):
        return None
    match = _AMBIGUOUS_NUMBER_CORRECTION.match(piece)
    if match is None:
        return None
    value = _parse_number(match.group("number"))
    if value is None or value <= 0:
        return None
    currency = match.groupdict().get("currency")
    formatted = _format_number(value)
    if currency:
        surface = re.sub(r"\s+", " ", currency).strip()
        return f"{formatted} {surface}"
    return formatted


def _parse_clause(sentence: str) -> _Change | None:
    piece = _strip_cues(sentence)
    if _is_empty_rejection(piece):
        return None
    change = _Change()
    found = False
    quantity = _QUANTITY_IS.search(piece)
    if quantity is not None:
        value = _parse_number(quantity.group("number"))
        if value is not None and value > 0:
            change.quantity = value
            found = True
    price = _PRICE_IS.search(piece) or _THEY_WERE_PRICE.search(piece)
    if price is not None:
        found = _read_money(change, price, price_scope=True) or found
    amount = _AMOUNT_IS.search(piece)
    if amount is not None and change.money is None:
        found = _read_money(change, amount, price_scope=False) or found
    item = _ITEM_IS.search(piece)
    if item is not None:
        change.item = item.group("item")
        found = True
    if _read_date(change, piece):
        found = True
    if found:
        return change
    return _parse_restatement(piece)


def _read_money(change: _Change, match: re.Match[str], *, price_scope: bool) -> bool:
    value = _parse_number(match.group("number"))
    if value is None or value <= 0:
        return False
    change.money = value
    currency = match.groupdict().get("currency")
    if currency:
        change.currency_surface = re.sub(r"\s+", " ", currency).strip()
    scope = match.groupdict().get("scope") or ""
    normalized = re.sub(r"\s+", " ", scope).strip().lower()
    if price_scope and normalized in {"total", "in total", "altogether"}:
        change.scope = "total"
    elif price_scope and normalized:
        change.scope = "unit"
    elif not price_scope:
        change.scope = "total"
    return True


def _read_date(change: _Change, piece: str) -> bool:
    match = _DATE_IS.search(piece)
    if match is None:
        return False
    rest = match.group("rest")
    concrete, invalid, _spans = find_concrete_dates(rest)
    if concrete is not None and not invalid:
        change.date = concrete
        change.relative_phrase = None
        change.clear_date = False
        return True
    relative = RELATIVE_DATE_RE.search(rest)
    if relative is not None:
        change.relative_phrase = relative.group(0)
        change.date = None
        change.clear_date = False
        return True
    if _OMITTED_DATE_RE.search(rest) is not None:
        # today and tonight are omitted, including when they replace a date.
        change.clear_date = True
        change.date = None
        change.relative_phrase = None
        return True
    return False


def _parse_restatement(piece: str) -> _Change | None:
    normalized = _AND_PRICE.sub(lambda match: f" for {match.group('price')}", piece)
    normalized = normalized.strip()
    if _TRADE_VERB.search(normalized) is None:
        return None
    evidence = analyze(normalized)
    if evidence.speech not in {"sale", "purchase", "expense"}:
        return None
    change = _Change()
    if evidence.quantity is not None and evidence.quantity > 0:
        change.quantity = evidence.quantity
    if evidence.money is not None and evidence.money > 0 and not evidence.vague_amount:
        change.money = evidence.money
        if evidence.amount_scope in {"unit", "total"}:
            change.scope = evidence.amount_scope
        if evidence.currency_surface:
            change.currency_surface = evidence.currency_surface
    # "I sold 20 for 2000 USD each" names no item. The leftover "20" must
    # not replace the item already stated in the original sentence.
    if (
        evidence.item
        and _NUMERIC_ITEM.fullmatch(evidence.item) is None
        and evidence.item.lower() in normalized.lower()
    ):
        change.item = evidence.item
    if evidence.date and not evidence.relative_date:
        change.date = evidence.date
    elif evidence.relative_date:
        relative = RELATIVE_DATE_RE.search(normalized)
        if relative is not None:
            change.relative_phrase = relative.group(0)
    elif _OMITTED_DATE_RE.search(normalized) is not None:
        change.clear_date = True
    change.utterance = normalized
    if not change.any():
        return None
    return change


def _rewrite(base: str, change: _Change) -> str | None:
    evidence = analyze(base) if base.strip() else Evidence(original=base)
    restated = analyze(change.utterance) if change.utterance else None
    replace_sentence = change.utterance is not None and (
        not base.strip()
        or evidence.speech == "unknown"
        or (
            restated is not None
            and restated.speech not in {"unknown", evidence.speech}
        )
    )
    if replace_sentence:
        text = change.utterance or ""
        if restated is not None:
            text = _splice(text, restated, change)
        if (
            base.strip()
            and not change.date
            and not change.relative_phrase
            and not change.clear_date
        ):
            text = text.rstrip(".!? ") + _date_suffix(base)
    elif not base.strip():
        return None
    else:
        text = _splice(base, evidence, change)
    if change.date or change.relative_phrase or change.clear_date:
        text = _apply_date(text, change)
    return _period(text)


def _splice(base: str, evidence: Evidence, change: _Change) -> str:
    text = base.strip()
    spans: list[tuple[int, int, str]] = []
    if change.quantity is not None:
        found = _role_span(text, "quantity")
        if found is not None:
            spans.append((found[0], found[1], _format_number(change.quantity)))
        elif evidence.item:
            text = re.sub(
                rf"\b({re.escape(evidence.item)})\b",
                f"{_format_number(change.quantity)} \\1",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
    pending_price: str | None = None
    if change.money is not None:
        found = _role_span(text, "money")
        if found is not None:
            spans.extend(_money_replacements(text, found, change, evidence))
        else:
            pending_price = _money_phrase(change)
    for start, end, replacement in sorted(spans, key=lambda item: item[0], reverse=True):
        text = text[:start] + replacement + text[end:]
    if pending_price is not None:
        text = _attach_price(text, pending_price)
    if change.item and evidence.item and change.item.lower() != evidence.item.lower():
        text = re.sub(
            rf"\b{re.escape(evidence.item)}\b",
            change.item,
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    if change.scope:
        text = _force_scope(text, change.scope)
    return text


def _money_replacements(
    text: str,
    number_span: tuple[int, int],
    change: _Change,
    evidence: Evidence,
) -> list[tuple[int, int, str]]:
    """Replace the amount and, when the correction names one, its currency.

    The number and the currency word are separate spans. Replacing only the
    number leaves the old currency in the sentence, and that word wins.
    """
    number = _format_number(change.money or Decimal(0))
    stated = change.currency_surface
    new_code = normalize_currency(stated) if stated else None
    if new_code is None or new_code == evidence.currency:
        return [(number_span[0], number_span[1], number)]
    surface = re.sub(r"\s+", " ", stated or "").strip()
    currency_span = _adjacent_currency_span(text, number_span)
    if currency_span is None:
        return [(number_span[0], number_span[1], f"{number} {surface}")]
    return [
        (number_span[0], number_span[1], number),
        (currency_span[0], currency_span[1], surface),
    ]


def _adjacent_currency_span(
    text: str,
    number_span: tuple[int, int],
) -> tuple[int, int] | None:
    """Currency word touching this amount, after it or before it."""
    start, end = number_span
    window = 40
    after_text = text[end : end + window]
    after = CURRENCY_RE.search(after_text)
    if after is not None and after_text[: after.start()].strip() == "":
        return end + after.start(), end + after.end()
    origin = max(0, start - window)
    before_text = text[origin:start]
    before = None
    for match in CURRENCY_RE.finditer(before_text):
        before = match
    if before is not None and before_text[before.end() :].strip() == "":
        return origin + before.start(), origin + before.end()
    return None


def _money_phrase(change: _Change) -> str:
    parts = [_format_number(change.money or Decimal(0))]
    if change.currency_surface:
        parts.append(change.currency_surface)
    if change.scope == "unit":
        parts.append("each")
    elif change.scope == "total":
        parts.append("total")
    return " ".join(parts)


def _attach_price(original: str, phrase: str) -> str:
    base = original.strip().rstrip(".!?")
    if re.search(r"\b(?:for|at)\b", base, re.IGNORECASE):
        return f"{base} {phrase}."
    return f"{base} for {phrase}."


def _force_scope(text: str, scope: str) -> str:
    if scope == "unit":
        text = re.sub(r"\b(?:in\s+total|altogether|total)\b", "", text, flags=re.IGNORECASE)
        if UNIT_CUE_RE.search(text) is None:
            text = text.rstrip(" .") + " each"
    elif scope == "total":
        text = re.sub(
            r"\b(?:each(?:\s+one)?|apiece)\b|\ba\s+piece\b(?!\s+of\b)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        if TOTAL_RE.search(text) is None:
            text = text.rstrip(" .") + " total"
    return re.sub(r"\s{2,}", " ", text).strip()


def _apply_date(text: str, change: _Change) -> str:
    text = _strip_dates(text)
    text = text.rstrip(" .")
    text = re.sub(r"\bon$", "", text, flags=re.IGNORECASE).strip()
    if change.date:
        return f"{text} on {change.date}"
    if change.relative_phrase:
        return f"{text} {change.relative_phrase}"
    return text


def _strip_dates(text: str) -> str:
    text = re.sub(
        rf"\b(?:on|dated)\s+{ISO_DATE_RE.pattern}",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(?:on|dated)\s+{MONTH_FIRST_RE.pattern}",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(?:on|dated)\s+{DAY_FIRST_RE.pattern}",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = ISO_DATE_RE.sub(" ", text)
    text = MONTH_FIRST_RE.sub(" ", text)
    text = DAY_FIRST_RE.sub(" ", text)
    text = RELATIVE_DATE_RE.sub(" ", text)
    text = _OMITTED_DATE_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _role_span(text: str, role: str) -> tuple[int, int] | None:
    occupied = _date_spans(text)
    for match in NUMBER_RE.finditer(text):
        if _covered(match.span(), occupied):
            continue
        after = text[match.end() : match.end() + 32]
        before = text[max(0, match.start() - 12) : match.start()]
        money = _is_money(before, after)
        if role == "money" and money:
            return match.span()
        if role == "quantity" and not money and re.match(r"\s+[A-Za-z]", after):
            return match.span()
    return None


def _date_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in (ISO_DATE_RE, MONTH_FIRST_RE, DAY_FIRST_RE):
        spans.extend(match.span() for match in pattern.finditer(text))
    return spans


def _covered(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(left <= start and end <= right for left, right in occupied)


def _strip_cues(sentence: str) -> str:
    piece = sentence.strip().rstrip(".!? ")
    previous = None
    while piece != previous:
        previous = piece
        piece = _CUE.sub("", piece, count=1).strip()
    return piece


def _is_empty_rejection(piece: str) -> bool:
    stripped = piece.strip().rstrip(".!? ")
    return not stripped or _EMPTY_REJECTION.match(stripped) is not None


def _date_suffix(base: str) -> str:
    concrete, invalid, _spans = find_concrete_dates(base)
    if concrete is not None and not invalid:
        return f" on {concrete}"
    relative = RELATIVE_DATE_RE.search(base)
    if relative is not None:
        return f" {relative.group(0)}"
    omitted = _OMITTED_DATE_RE.search(base)
    if omitted is not None:
        return f" {omitted.group(0)}"
    return ""


def _format_number(value: Decimal) -> str:
    integral = value.to_integral_value()
    if value == integral:
        return str(int(integral))
    return format(value, "f")


def _period(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" ,")
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
    cleaned = cleaned.rstrip(".!? ")
    return f"{cleaned}."
