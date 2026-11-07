"""Combine a caller-supplied clarification with the user's answer.

The service stays stateless. Nothing here is stored. The caller either sends
``context`` or the single text block the Meri frontend already builds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.interpreter.corrections import partition_corrections, revise_corrections
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
    r"^(?:how\b|what(?:'s|s)?\b|who\b|where\b|when\b|which\b|why\b|"
    r"(?:did|do|have|am|is|are)\s+(?:i|we|you|there)\b|"
    r"(?:i|we)\s+(?:just\s+|have\s+|had\s+)?"
    r"(?:sold|bought|spent|lost|found|owe|paid|purchased)\b|"
    r"(?:record|add)\s+|"
    r"(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|"
    r"thanks|thank\s+you|yo|sup)\b)",
    re.IGNORECASE,
)
_CONVERSATIONAL_CHITCHAT_RE = re.compile(
    r"^(?:what(?:'s|s|\s+is)\s+up\b|how\s+are\s+(?:you|u)\b|"
    r"how(?:'s|s|\s+is)\s+it\s+going\b|who\s+are\s+(?:you|u)\b|"
    r"what\s+can\s+you\s+do\b|are\s+you\s+there\b|"
    r"hi\b|hello\b|hey\b|good\s+(?:morning|afternoon|evening)\b|"
    r"thanks\b|thank\s+you\b|yo\b|sup\b|"
    r"never\s*mind\b|cancel\b|forget\s+it\b)",
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
class ResolvedUtterance:
    """One utterance, or a correction that named no replacement."""

    text: str
    unresolved_correction: bool = False
    ambiguous_number: str | None = None
    unrelated_message: bool = False


@dataclass(frozen=True)
class _ComposeResult:
    text: str
    unresolved: bool = False
    ambiguous_number: str | None = None
    unrelated: bool = False


@dataclass(frozen=True)
class _ParsedContinuation:
    original: str
    answer: str
    missing_fields: tuple[str, ...]


def resolve_utterance(text: str, continuation: Continuation | None = None) -> ResolvedUtterance:
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
        composed = _compose(original, answer, missing)
    elif parsed is not None:
        composed = _compose(parsed.original, parsed.answer, parsed.missing_fields)
    else:
        composed = _ComposeResult(normalized)
    if composed.unrelated:
        return ResolvedUtterance(composed.text, unrelated_message=True)
    if composed.unresolved:
        return ResolvedUtterance(
            composed.text,
            unresolved_correction=True,
            ambiguous_number=composed.ambiguous_number,
        )
    # A raw utterance can carry the correction in a later sentence.
    revised = revise_corrections(composed.text)
    return ResolvedUtterance(
        revised.text,
        unresolved_correction=revised.unresolved,
        ambiguous_number=revised.ambiguous_number,
    )


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


def _compose(original: str, answer: str, missing: tuple[str, ...]) -> _ComposeResult:
    original = original.strip()
    answer = answer.strip()
    if not answer:
        return _ComposeResult(original)
    if _is_unrelated_conversational(answer):
        return _ComposeResult(answer, unrelated=True)
    if _is_new_request(answer):
        # A pending clarification must not reclassify a complete new utterance.
        # "I just sold ..." does not match the frontend's "i sold " prefix, so
        # the frontend wraps it. The answer itself is the request.
        return _ComposeResult(answer)
    if not original:
        return _ComposeResult(answer)

    # Peel a correction before a price answer is glued onto the original.
    # Otherwise "price is 2000" becomes another number beside the old price.
    head, clauses = partition_corrections(answer)
    if clauses:
        if head:
            nested = _compose(original, head, missing)
            if nested.unresolved or nested.unrelated:
                return nested
            base = nested.text
        else:
            base = original
        revised = revise_corrections(_join_clauses(base, clauses))
        return _ComposeResult(
            revised.text,
            unresolved=revised.unresolved,
            ambiguous_number=revised.ambiguous_number,
        )

    scope = _SCOPE_ANSWER_RE.match(answer)
    if scope is not None:
        return _ComposeResult(_ensure_cue(original, _scope_cue(scope.group("cue"))))

    if analyze(answer).money is None:
        base = original.rstrip().rstrip(".!?")
        return _ComposeResult(f"{base}. {answer.rstrip('.!?')}.")

    phrase = _LEADING_PRICE_RE.sub("", answer).rstrip(".!? ")
    has_unit = UNIT_CUE_RE.search(phrase) is not None
    has_total = TOTAL_RE.search(phrase) is not None
    if not has_unit and not has_total and _price_is_the_total(original, missing):
        phrase = f"{phrase} total"
        has_total = True
    if _same_price(original, phrase):
        if has_unit:
            return _ComposeResult(_ensure_cue(original, "each"))
        if has_total:
            return _ComposeResult(_ensure_cue(original, "total"))
        finished = original if original.endswith(".") else f"{original.rstrip('.!?')}."
        return _ComposeResult(finished)
    return _ComposeResult(_attach_price(original, phrase))


def _join_clauses(base: str, clauses: list[str]) -> str:
    prefix = base.strip()
    tail = " ".join(clause.strip() for clause in clauses)
    if not prefix:
        return tail
    if prefix[-1] not in ".!?":
        prefix += "."
    return f"{prefix} {tail}"


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


def _is_unrelated_conversational(answer: str) -> bool:
    return _CONVERSATIONAL_CHITCHAT_RE.match(answer.strip()) is not None


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
