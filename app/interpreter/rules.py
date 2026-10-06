"""English patterns for the cases the service must get right without guessing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from app.interpreter.normalization import (
    MAX_QUANTITY,
    currency_display,
    find_concrete_dates,
    mask_spans,
    normalize_currency,
)
from app.interpreter.raw import Draft

WORD_NUMBERS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_NUMBER_WORDS = "|".join(WORD_NUMBERS)
NUMBER_RE = re.compile(
    rf"\b(\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?|{_NUMBER_WORDS})\b",
    re.IGNORECASE,
)
CURRENCY_RE = re.compile(
    r"\b(ethiopian\s+birr|birr|etb|us\s+dollars?|dollars?|usd|euros?|eur|pounds?|gbp)\b",
    re.IGNORECASE,
)
# "per" is a unit price only in "<price> per <item>". "as per the receipt" is not.
UNIT_CUE_RE = re.compile(
    r"\b(?:each(?:\s+one)?|apiece|a\s+piece\b(?!\s+of\b)|"
    r"(?<!as\s)per\s+(?!the\b|our\b|a\b|an\b|this\b|that\b|my\b|your\b)\w+)\b",
    re.IGNORECASE,
)
TOTAL_RE = re.compile(r"\b(in total|altogether|total)\b", re.IGNORECASE)
_RELATIVE_COUNT = (
    r"(?:a\s+few|couple\s+of|several|an|a|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|\d+)"
)
RELATIVE_DATE_RE = re.compile(
    rf"\b(?:yesterday|today|tomorrow|tonight|this morning|last night|"
    rf"(?:this|last|next)\s+(?:week|month|year)|"
    rf"{_RELATIVE_COUNT}\s+(?:day|week|month|year)s?\s+ago)\b",
    re.IGNORECASE,
)
_LOSS_VERB_RE = re.compile(
    r"\b(?:lost|damaged|stolen|stole|spoiled|spoil|spoilage|removed|remove|"
    r"broke|broken|break)\b",
    re.IGNORECASE,
)
_GAIN_VERB_RE = re.compile(r"\bfound\b", re.IGNORECASE)
_INVENTORY_VERBS = (
    "lost|found|damaged|stolen|stole|spoiled|spoil|spoilage|removed|remove|"
    "broke|broken|break"
)
_HAD_ITEM_LOSS_RE = re.compile(
    r"\b(?:had|have)\s+(?P<body>.+?)\s+"
    r"(?:stolen|damaged|spoiled|lost|broken|removed)\b",
    re.IGNORECASE,
)
_SCOPE_QUESTION_RE = re.compile(
    r"the total for all\b.+\bor\b.+\bper\b",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_SCOPE_ANSWER_RE = re.compile(
    r"(?:^|[.!?]\s+)(?P<answer>each(?:\s+one)?|apiece|a\s+piece|per\s+[a-z]+|"
    r"total|in\s+total|altogether)\s*[.!?]*$",
    re.IGNORECASE,
)
VAGUE_ITEMS = {"something", "anything", "stuff", "things", "thing", "whatever"}
STOP_NAMES = {
    "the",
    "a",
    "an",
    "my",
    "our",
    "him",
    "her",
    "them",
    "me",
    "you",
    "us",
    "some",
    "someone",
    "somebody",
    "who",
    "what",
    "that",
    "he",
    "she",
    "they",
    "store",
    "shop",
}
_LEAD_RE = re.compile(
    rf"^(?P<lead>a few|a lot of|lots of|some|many|several|an|a|{_NUMBER_WORDS}|"
    r"\d[\d,]*(?:\.\d+)?)\s+(?P<rest>.+)$",
    re.IGNORECASE,
)
_ITEM_SPLIT_RE = re.compile(
    r"\s+\b(?:for|at|to|from|on|yesterday|today|tomorrow|tonight)\b",
    re.IGNORECASE,
)
_INTERROGATIVE_RE = re.compile(
    r"^(how many|how much|how|what|who|when|which|where|why|did i|do i|have i|"
    r"am i|is there|are there|what's|whats)\b",
    re.IGNORECASE,
)
_ASSERTION_RE = re.compile(
    r"^(i|we)\s+(sold|bought|spent|lost|found|owe|paid|purchased)\b",
    re.IGNORECASE,
)


@dataclass
class Evidence:
    original: str
    speech: str = "unknown"
    item: str | None = None
    description: str | None = None
    customer: str | None = None
    supplier: str | None = None
    reason: str | None = None
    direction: str | None = None
    quantity: Decimal | None = None
    quantity_surface: str | None = None
    money: Decimal | None = None
    money_surface: str | None = None
    currency: str | None = None
    currency_surface: str | None = None
    currency_unrecognized: bool = False
    amount_scope: str = "absent"
    date: str | None = None
    date_invalid: bool = False
    relative_date: bool = False
    vague_quantity: bool = False
    vague_amount: bool = False
    vague_item: bool = False
    numbers: list[Decimal] = field(default_factory=list)
    money_values: list[Decimal] = field(default_factory=list)
    quantity_values: list[Decimal] = field(default_factory=list)


def is_query(text: str) -> bool:
    stripped = text.strip()
    if _ASSERTION_RE.match(stripped):
        return False
    if stripped.endswith("?"):
        return True
    return _INTERROGATIVE_RE.match(stripped) is not None


def analyze(text: str) -> Evidence:
    original = text.strip()
    evidence = Evidence(original=original)
    if not original:
        return evidence
    if is_query(original):
        evidence.speech = "query"
        return evidence

    working = original.rstrip(".!?")
    concrete, invalid, spans = find_concrete_dates(working)
    relative_spans = [match.span() for match in RELATIVE_DATE_RE.finditer(working)]
    evidence.date_invalid = invalid
    # A relative date the service cannot normalize must not be dropped.
    if relative_spans:
        evidence.relative_date = True
        evidence.date = None
    else:
        evidence.date = concrete
    masked = mask_spans(working, spans + relative_spans)

    _extract_currency(masked, evidence)
    _extract_numbers(masked, evidence)
    _extract_vague(masked, evidence)
    _extract_speech(working, evidence)
    _extract_scope(working, evidence)

    if evidence.speech == "customer_debt":
        _extract_debt(working, evidence)
    elif evidence.speech == "inventory_adjustment":
        _extract_inventory(working, evidence)
    elif evidence.speech == "expense":
        _extract_expense(working, evidence)
    elif evidence.speech in {"sale", "purchase"}:
        _extract_trade(working, evidence)
    return evidence


def draft_from_evidence(evidence: Evidence) -> Draft:
    if evidence.speech == "query":
        return Draft(kind="query", confident=True, query=evidence.original)
    if evidence.speech == "unknown":
        return Draft(kind="clarification", confident=False)

    amount: Decimal | None = None
    unit_price: Decimal | None = None
    if evidence.money is not None and not evidence.vague_amount:
        if evidence.amount_scope == "unit":
            unit_price = evidence.money
        elif evidence.amount_scope != "ambiguous":
            amount = evidence.money

    quantity = None if evidence.vague_quantity else evidence.quantity
    item = None if evidence.vague_item else evidence.item
    date = evidence.date if evidence.date and not evidence.relative_date else None
    return Draft(
        kind="create_event",
        confident=True,
        event_type=evidence.speech,
        item=item,
        quantity=quantity,
        amount=amount,
        unit_price=unit_price,
        amount_scope=evidence.amount_scope,
        currency=evidence.currency if evidence.money is not None else None,
        customer=evidence.customer,
        supplier=evidence.supplier,
        description=evidence.description,
        reason=evidence.reason,
        direction=evidence.direction,
        date=date,
        vague_quantity=evidence.vague_quantity,
        vague_amount=evidence.vague_amount,
        vague_item=evidence.vague_item,
    )


def _extract_currency(text: str, evidence: Evidence) -> None:
    match = CURRENCY_RE.search(text)
    if match is None:
        return
    surface = re.sub(r"\s+", " ", match.group(1)).strip()
    code = normalize_currency(surface)
    if code is None:
        evidence.currency_unrecognized = True
        evidence.currency_surface = surface
        return
    evidence.currency = code
    evidence.currency_surface = currency_display(surface, code)


def _extract_numbers(text: str, evidence: Evidence) -> None:
    quantity: tuple[Decimal, str] | None = None
    money: tuple[Decimal, str, str] | None = None
    for match in NUMBER_RE.finditer(text):
        value = _parse_number(match.group(1))
        if value is None or abs(value) > MAX_QUANTITY:
            continue
        evidence.numbers.append(value)
        after = text[match.end() : match.end() + 32]
        before = text[max(0, match.start() - 12) : match.start()]
        if _is_money(before, after):
            evidence.money_values.append(value)
            if money is None:
                money = (value, match.group(1), after)
            continue
        if re.match(r"\s+[A-Za-z]", after):
            evidence.quantity_values.append(value)
            if quantity is None:
                quantity = (value, match.group(1))
    if quantity is not None:
        evidence.quantity = quantity[0]
        evidence.quantity_surface = quantity[1]
    if money is not None:
        evidence.money = money[0]
        evidence.money_surface = money[1]
        _note_unknown_currency(money[2], evidence)


def _is_money(before: str, after: str) -> bool:
    if re.match(rf"\s+{CURRENCY_RE.pattern}", after, re.IGNORECASE):
        return True
    if CURRENCY_RE.match(after.strip()):
        return True
    if re.search(r"\b(for|at)\s*$", before, re.IGNORECASE):
        return True
    return re.match(r"\s+(each|apiece)\b", after, re.IGNORECASE) is not None


def _note_unknown_currency(after: str, evidence: Evidence) -> None:
    if evidence.currency is not None or evidence.currency_unrecognized:
        return
    match = re.match(r"\s+([A-Za-z]+)", after)
    if match is None:
        return
    word = match.group(1)
    if word.lower() in {
        "total",
        "each",
        "apiece",
        "piece",
        "one",
        "per",
        "in",
        "on",
        "for",
        "to",
        "from",
        "and",
        "the",
    }:
        return
    if normalize_currency(word) is None:
        evidence.currency_unrecognized = True
        evidence.currency_surface = word


def _parse_number(surface: str) -> Decimal | None:
    token = surface.strip().lower().replace(",", "")
    if token in WORD_NUMBERS:
        return Decimal(WORD_NUMBERS[token])
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return value


def _extract_vague(text: str, evidence: Evidence) -> None:
    # "some money" is a vague price, not a vague count.
    if re.search(r"\b(a few|a lot of|lots of|many|several|a lot|lots)\b", text, re.IGNORECASE):
        evidence.vague_quantity = True
    elif re.search(r"\bsome\b", text, re.IGNORECASE) and not re.search(
        r"\bsome\s+(money|cash)\b", text, re.IGNORECASE
    ):
        evidence.vague_quantity = True
    has_money = evidence.money is not None
    if not has_money and re.search(
        r"\b(some money|some cash|money|cash|something|anything)\b",
        text,
        re.IGNORECASE,
    ):
        evidence.vague_amount = True


def inventory_polarity(text: str) -> str | None:
    """Return loss, gain, ambiguous, or None from the source text alone."""
    loss = _LOSS_VERB_RE.search(text) is not None
    gain = _GAIN_VERB_RE.search(text) is not None
    if loss and gain:
        return "ambiguous"
    if loss:
        return "loss"
    if gain:
        return "gain"
    return None


def inventory_reason(text: str) -> str | None:
    """Reason word for a clear inventory verb. None when the direction is mixed."""
    if inventory_polarity(text) == "ambiguous":
        return None
    if re.search(r"\bdamaged\b", text, re.IGNORECASE):
        return "damaged"
    if re.search(r"\b(?:stolen|stole)\b", text, re.IGNORECASE):
        return "stolen"
    if re.search(r"\b(?:spoiled|spoil|spoilage)\b", text, re.IGNORECASE):
        return "spoiled"
    if re.search(r"\b(?:removed|remove)\b", text, re.IGNORECASE):
        return "removed"
    if re.search(r"\b(?:broke|broken|break)\b", text, re.IGNORECASE):
        return "broken"
    if re.search(r"\blost\b", text, re.IGNORECASE):
        return "lost"
    if re.search(r"\bfound\b", text, re.IGNORECASE):
        return "found"
    return None


def _extract_speech(text: str, evidence: Evidence) -> None:
    if re.search(r"\b(owe|owes|owed)\b", text, re.IGNORECASE):
        evidence.speech = "customer_debt"
    elif inventory_polarity(text) is not None:
        evidence.speech = "inventory_adjustment"
    elif re.search(r"\b(spent|spend|spending|expense)\b", text, re.IGNORECASE):
        evidence.speech = "expense"
    elif re.search(r"\b(bought|buy|buying|purchased|purchase)\b", text, re.IGNORECASE):
        evidence.speech = "purchase"
    elif re.search(r"\b(sold|sell|selling|sale)\b", text, re.IGNORECASE):
        evidence.speech = "sale"
    else:
        evidence.speech = "unknown"


def _scope_text(text: str) -> str:
    """Use a trailing Each/Total answer when the text replays the scope question.

    The question itself contains both "total" and "per", so those words must not
    compete with the user's answer. Ordinary sentences are left unchanged.
    """
    if _SCOPE_QUESTION_RE.search(text) is None:
        return text
    match = _TRAILING_SCOPE_ANSWER_RE.search(text.strip())
    if match is None:
        return text
    return match.group("answer")


def _extract_scope(text: str, evidence: Evidence) -> None:
    scoped = _scope_text(text)
    has_each = UNIT_CUE_RE.search(scoped) is not None
    has_total = TOTAL_RE.search(scoped) is not None
    trade = evidence.speech in {"sale", "purchase", "unknown"}
    if evidence.speech in {"expense", "customer_debt", "inventory_adjustment", "query"}:
        evidence.amount_scope = "absent"
        return
    if not trade:
        evidence.amount_scope = "absent"
        return
    if has_each and has_total:
        evidence.amount_scope = "ambiguous"
    elif has_each:
        evidence.amount_scope = "unit"
    elif has_total:
        evidence.amount_scope = "total"
    elif (
        evidence.quantity is not None
        and evidence.money is not None
        and not evidence.vague_quantity
    ):
        evidence.amount_scope = "ambiguous"
    else:
        evidence.amount_scope = "absent"


def _extract_trade(text: str, evidence: Evidence) -> None:
    verbs = "sold|sell|selling|sale|bought|buy|buying|purchased|purchase"
    item, vague_item = _item_after(text, verbs)
    evidence.item = item
    evidence.vague_item = vague_item or evidence.vague_item
    if evidence.speech == "sale":
        evidence.customer = _named_party(text, "to")
    else:
        evidence.supplier = _named_party(text, "from")


def _extract_inventory(text: str, evidence: Evidence) -> None:
    had = _HAD_ITEM_LOSS_RE.search(text)
    if had is not None:
        item, vague_item = _item_from_body(had.group("body"))
    else:
        item, vague_item = _item_after(text, _INVENTORY_VERBS)
    evidence.item = item
    evidence.vague_item = vague_item or evidence.vague_item
    polarity = inventory_polarity(text)
    evidence.reason = inventory_reason(text)
    if polarity != "loss" and polarity != "gain":
        evidence.quantity = None
        return
    if evidence.vague_quantity:
        evidence.quantity = None
        return
    if evidence.quantity is not None:
        sign = -1 if polarity == "loss" else 1
        evidence.quantity = evidence.quantity.copy_abs() * sign


def _extract_expense(text: str, evidence: Evidence) -> None:
    match = re.search(r"\b(?:on|for)\s+(.+)$", text, re.IGNORECASE)
    if match is None:
        return
    description = RELATIVE_DATE_RE.split(match.group(1))[0]
    description = _clean_phrase(description)
    if not description or description.lower() in VAGUE_ITEMS | {"money", "cash", "it"}:
        evidence.description = None
        return
    evidence.description = description


def _extract_debt(text: str, evidence: Evidence) -> None:
    owes_me = re.search(r"\b([A-Za-z][\w'-]*)\s+owes\s+me\b", text, re.IGNORECASE)
    i_owe = re.search(r"\bI\s+owe\s+([A-Za-z][\w'-]*)\b", text, re.IGNORECASE)
    owes = re.search(r"\b([A-Za-z][\w'-]*)\s+owes\b", text, re.IGNORECASE)
    if owes_me is not None and owes_me.group(1).lower() not in STOP_NAMES:
        evidence.customer = owes_me.group(1)
        evidence.direction = "owed_to_business"
    elif i_owe is not None and i_owe.group(1).lower() not in STOP_NAMES:
        evidence.customer = i_owe.group(1)
        evidence.direction = "owed_by_business"
    elif owes is not None and owes.group(1).lower() not in STOP_NAMES | {"i"}:
        evidence.customer = owes.group(1)
        evidence.direction = None


def _item_after(text: str, verbs: str) -> tuple[str | None, bool]:
    match = re.search(rf"\b(?:{verbs})\s+(?P<body>.*)$", text, re.IGNORECASE)
    if match is None:
        return None, False
    return _item_from_body(match.group("body"))


def _item_from_body(body: str) -> tuple[str | None, bool]:
    body = _ITEM_SPLIT_RE.split(body, maxsplit=1)[0]
    body = re.sub(r"^(?:of|the)\s+", "", body.strip(), flags=re.IGNORECASE)
    body = _clean_phrase(RELATIVE_DATE_RE.sub(" ", body))
    if not body:
        return None, False
    if body.lower() in VAGUE_ITEMS:
        return None, True
    lead = _LEAD_RE.match(body)
    if lead is None:
        return body, False
    rest = _clean_phrase(lead.group("rest"))
    if rest.lower() in VAGUE_ITEMS:
        return None, True
    return (rest or None), False


def _named_party(text: str, preposition: str) -> str | None:
    match = re.search(rf"\b{preposition}\s+([A-Za-z][\w'-]*)", text, re.IGNORECASE)
    if match is None:
        return None
    name = match.group(1)
    if name.lower() in STOP_NAMES:
        return None
    return name


def _clean_phrase(value: str) -> str:
    cleaned = value.strip(" \t\r\n.,!?:;\"'")
    return re.sub(r"\s+", " ", cleaned)
