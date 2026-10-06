"""Combine a caller-supplied clarification with the user's answer.

The service stays stateless. Nothing here is stored. The caller either sends
``context`` or the single text block the Meri frontend already builds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.interpreter.rules import TOTAL_RE, UNIT_CUE_RE, analyze

# The production frontend posts one string. It has no context object.
_FRONTEND_CONTINUATION_RE = re.compile(
    r"\A\s*Original user statement:\s*(?P<original>\S.*?)\n"
    r"\s*Clarification question:\s*(?P<question>.*?)\n"
    r"\s*Missing fields:\s*(?P<fields>.*?)\n"
    r"\s*User clarification answer:\s*(?P<answer>.*?)\n"
    r"\s*Resolve the original request using the clarification answer\.?\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_SCOPE_ANSWER_RE = re.compile(
    r"^(?:it(?:'s| is)\s+)?(?P<cue>each(?:\s+one)?|apiece|a\s+piece|per\s+\w+|"
    r"the\s+total|in\s+total|altogether|total|for\s+all(?:\s+of\s+them)?|"
    r"all\s+of\s+them)\s*[.!?]*$",
    re.IGNORECASE,
)
_NEW_REQUEST_RE = re.compile(
    r"^(?:how |what |who |where |when |which |"
    r"(?:i|we)\s+(?:just\s+|have\s+|had\s+)?"
    r"(?:sold|bought|spent|lost|found|owe|paid|purchased)|"
    r"(?:record|add)\s+)",
    re.IGNORECASE,
)
_LEADING_PRICE_RE = re.compile(r"^(?:for|at)\s+", re.IGNORECASE)
_TOTAL_CUES = {
    "total",
    "the total",
    "in total",
    "altogether",
    "for all",
    "for all of them",
    "all of them",
}


@dataclass(frozen=True)
class Continuation:
    """Original utterance supplied by the caller for this one request."""

    original_text: str
    missing_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ParsedContinuation:
    original: str
    answer: str
    missing_fields: tuple[str, ...]


def resolve_utterance(text: str, continuation: Continuation | None = None) -> str:
    """Return one utterance the existing rules can interpret."""
    normalized = text.replace("\r\n", "\n").strip()
    parsed = _parse_frontend_continuation(normalized)
    if continuation is not None and continuation.original_text.strip():
        original = continuation.original_text.strip()
        missing = continuation.missing_fields
        answer = normalized
        if parsed is not None:
            answer = parsed.answer
            if not missing:
                missing = parsed.missing_fields
        return _compose(original, answer, missing)
    if parsed is not None:
        return _compose(parsed.original, parsed.answer, parsed.missing_fields)
    return normalized


def _parse_frontend_continuation(text: str) -> _ParsedContinuation | None:
    match = _FRONTEND_CONTINUATION_RE.match(text)
    if match is None:
        return None
    fields = tuple(
        field.strip()
        for field in match.group("fields").split(",")
        if field.strip()
    )
    return _ParsedContinuation(
        original=match.group("original").strip(),
        answer=match.group("answer").strip(),
        missing_fields=fields,
    )


def _compose(original: str, answer: str, missing: tuple[str, ...]) -> str:
    original = original.strip()
    answer = answer.strip()
    if not answer:
        return original
    if _is_new_request(answer):
        # A pending clarification must not reclassify a complete new utterance.
        # "I just sold ..." does not match the frontend's "i sold " prefix, so
        # the frontend wraps it. The answer itself is the request.
        return answer
    if not original:
        return answer

    scope = _SCOPE_ANSWER_RE.match(answer)
    if scope is not None:
        return _ensure_cue(original, _scope_cue(scope.group("cue")))

    if analyze(answer).money is None:
        base = original.rstrip().rstrip(".!?")
        return f"{base}. {answer.rstrip('.!?')}."

    phrase = _LEADING_PRICE_RE.sub("", answer).rstrip(".!? ")
    has_unit = UNIT_CUE_RE.search(phrase) is not None
    has_total = TOTAL_RE.search(phrase) is not None
    if not has_unit and not has_total and _price_is_the_total(original, missing):
        phrase = f"{phrase} total"
        has_total = True
    if _same_price(original, phrase):
        if has_unit:
            return _ensure_cue(original, "each")
        if has_total:
            return _ensure_cue(original, "total")
        return original if original.endswith(".") else f"{original.rstrip('.!?')}."
    return _attach_price(original, phrase)


def _price_is_the_total(original: str, missing: tuple[str, ...]) -> bool:
    """An amount supplied for the first time answers "how much was the sale?"

    A price already sitting next to a quantity stays ambiguous until the user
    says total or each. ``amount_scope`` in the missing fields means the same.
    """
    names = {field.strip().lower() for field in missing}
    if "amount_scope" in names:
        return False
    evidence = analyze(original)
    if evidence.money is None:
        return True
    return "amount" in names


def _is_new_request(answer: str) -> bool:
    return _NEW_REQUEST_RE.match(answer.strip()) is not None


def _scope_cue(cue: str) -> str:
    normalized = re.sub(r"\s+", " ", cue.strip().lower())
    if normalized in _TOTAL_CUES:
        return "total"
    if normalized in {"each", "each one"}:
        return "each"
    return normalized


def _ensure_cue(original: str, cue: str) -> str:
    base = original.strip().rstrip(".!?")
    if cue == "total" and TOTAL_RE.search(base):
        return f"{base}."
    if cue != "total" and UNIT_CUE_RE.search(base):
        return f"{base}."
    return f"{base} {cue}."


def _same_price(original: str, phrase: str) -> bool:
    original_money = analyze(original).money
    answer_money = analyze(phrase).money
    return original_money is not None and original_money == answer_money


def _attach_price(original: str, phrase: str) -> str:
    base = original.strip().rstrip(".!?")
    if re.search(r"\b(?:for|at)\b", base, re.IGNORECASE):
        return f"{base} {phrase}."
    return f"{base} for {phrase}."
