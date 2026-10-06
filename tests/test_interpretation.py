"""Behavior of the interpreter on the MVP utterances."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

import pytest

from app.interpreter.raw import Draft
from app.interpreter.rules import analyze
from app.interpreter.service import interpret_text
from app.interpreter.validation import decide, finalize
from app.llm.errors import ProviderUnavailable

SALE_TOTAL = "I sold five shirts for 900 birr total."
SALE_EACH = "I sold five shirts for 900 birr each."
SALE_AMBIGUOUS = "I sold five shirts for 900 birr."


def _data(text: str, provider=None) -> dict:
    result = interpret_text(text, "en", provider)
    return result.model_dump(mode="json", exclude_none=True)


def test_sale_total() -> None:
    assert _data(SALE_TOTAL) == {
        "type": "create_event",
        "event_type": "sale",
        "data": {"item": "shirts", "quantity": 5, "amount": 900, "currency": "ETB"},
    }


def test_sale_unit_price_is_multiplied() -> None:
    payload = _data(SALE_EACH)
    assert payload["data"]["amount"] == 4500
    assert isinstance(payload["data"]["amount"], int)
    assert payload["data"]["quantity"] == 5
    assert payload["event_type"] == "sale"


def test_sale_amount_scope_is_not_assumed() -> None:
    payload = _data(SALE_AMBIGUOUS)
    assert payload == {
        "type": "clarification",
        "question": "Is 900 birr the total for all five shirts, or 900 birr per shirt?",
        "missing_fields": ["amount_scope"],
    }


def test_sale_missing_amount() -> None:
    payload = _data("I sold five shirts.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["amount"]
    assert payload["question"] == "How much was the sale?"


def test_sale_missing_quantity() -> None:
    payload = _data("I sold shirts for 900 birr.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["quantity"]
    assert payload["question"] == "How many shirts did you sell?"


def test_sale_something_is_not_an_event() -> None:
    payload = _data("I sold something.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["item", "quantity", "amount"]
    assert "quantity" not in payload
    assert "amount" not in payload


def test_purchase_total() -> None:
    assert _data("I bought 20 shirts for 5000 birr total.")["data"] == {
        "item": "shirts",
        "quantity": 20,
        "amount": 5000,
        "currency": "ETB",
    }


def test_purchase_unit_price_is_multiplied() -> None:
    payload = _data("I bought 20 shirts for 250 birr each.")
    assert payload["event_type"] == "purchase"
    assert payload["data"]["amount"] == 5000
    assert payload["data"]["quantity"] == 20


def test_purchase_amount_scope_is_not_assumed() -> None:
    payload = _data("I bought 20 shirts for 250 birr.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["amount_scope"]
    assert "250" in payload["question"]
    assert "20" in payload["question"]


def test_expense() -> None:
    assert _data("I spent 300 birr on transportation.") == {
        "type": "create_event",
        "event_type": "expense",
        "data": {"description": "transportation", "amount": 300, "currency": "ETB"},
    }


def test_expense_missing_amount() -> None:
    payload = _data("I spent money on transportation.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["amount"]


def test_debt_owed_to_business() -> None:
    assert _data("John owes me 500 birr.")["data"] == {
        "customer": "John",
        "amount": 500,
        "direction": "owed_to_business",
        "currency": "ETB",
    }


def test_debt_owed_by_business() -> None:
    assert _data("I owe John 500 birr.")["data"]["direction"] == "owed_by_business"


def test_debt_missing_amount() -> None:
    payload = _data("John owes me money.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["amount"]
    assert "owed_to_business" not in payload["question"]


@pytest.mark.parametrize(
    "text",
    [
        "How much did I sell today?",
        "What did I spend this week?",
        "How many shirts do I have left?",
        "Who owes me money?",
    ],
)
def test_queries_keep_the_question(text: str) -> None:
    assert _data(text) == {"type": "query", "query": text}


@pytest.mark.parametrize(
    ("text", "required"),
    [
        ("I sold a few shirts for 900 birr.", ["quantity"]),
        ("I sold some shirts for 900 birr.", ["quantity"]),
        ("I sold five shirts for some money.", ["amount"]),
        ("I sold five shirts.", ["amount"]),
        ("I bought some things.", ["item", "quantity", "amount"]),
        ("I sold some shirts.", ["quantity"]),
        ("I bought some stuff.", ["item", "quantity", "amount"]),
        ("John owes me something.", ["amount"]),
    ],
)
def test_vague_values_are_not_fabricated(text: str, required: list[str]) -> None:
    payload = _data(text)
    assert payload["type"] == "clarification"
    assert "data" not in payload
    for name in required:
        assert name in payload["missing_fields"]
    source_numbers = set(re.findall(r"\d+", text))
    question_numbers = set(re.findall(r"\d+", payload["question"]))
    assert question_numbers <= source_numbers
    assert "4500" not in payload["question"]
    assert "5000" not in payload["question"]


def test_lost_shirts_are_a_negative_adjustment() -> None:
    assert _data("I lost 3 shirts.") == {
        "type": "create_event",
        "event_type": "inventory_adjustment",
        "data": {"item": "shirts", "quantity": -3, "reason": "lost"},
    }


def test_found_shirts_are_a_positive_adjustment() -> None:
    payload = _data("I found 3 shirts.")
    assert payload["data"] == {"item": "shirts", "quantity": 3, "reason": "found"}


def test_zero_inventory_is_not_recorded() -> None:
    payload = _data("I lost 0 shirts.")
    assert payload["type"] == "clarification"
    assert "quantity" in payload["missing_fields"]


def test_rules_do_not_call_the_provider_for_a_known_sale() -> None:
    class Boom:
        def interpret(self, text: str, language: str) -> Draft:
            raise AssertionError("provider should not be called")

    payload = _data(SALE_AMBIGUOUS, Boom())
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["amount_scope"]


def test_model_cannot_override_an_ambiguous_price() -> None:
    draft = Draft(
        kind="create_event",
        event_type="sale",
        item="shirts",
        quantity=Decimal(5),
        amount=Decimal("900"),
        currency="ETB",
        amount_scope="total",
    )
    payload = finalize(draft, SALE_AMBIGUOUS).model_dump(mode="json", exclude_none=True)
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["amount_scope"]
    assert "data" not in payload
    assert "4500" not in payload["question"]


def test_model_numbers_must_appear_in_the_text() -> None:
    class Inventive:
        def interpret(self, text: str, language: str) -> Draft:
            return Draft(
                kind="create_event",
                event_type="sale",
                item="coffees",
                quantity=9,
                amount=80,
                currency="ETB",
                amount_scope="total",
            )

    payload = _data("Client took 2 coffees for 40 birr altogether.", Inventive())
    assert payload == {
        "type": "create_event",
        "event_type": "sale",
        "data": {"item": "coffees", "quantity": 2, "amount": 40, "currency": "ETB"},
    }


def test_ambiguous_paraphrase_stays_a_clarification() -> None:
    class ForceTotal:
        def interpret(self, text: str, language: str) -> Draft:
            return Draft(
                kind="create_event",
                event_type="sale",
                item="coffees",
                quantity=2,
                amount=40,
                currency="ETB",
                amount_scope="total",
            )

    payload = _data("Client took 2 coffees for 40 birr.", ForceTotal())
    assert payload["type"] == "clarification"
    assert "amount_scope" in payload["missing_fields"]


def test_unknown_text_without_a_provider_asks_for_intent() -> None:
    payload = _data("Client took coffees home.")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["intent"]


def test_provider_failure_propagates() -> None:
    class Down:
        def interpret(self, text: str, language: str) -> Draft:
            raise ProviderUnavailable()

    with pytest.raises(ProviderUnavailable):
        interpret_text("Client took coffees home.", "en", Down())


def test_each_apiece_and_per_item_are_unit_prices() -> None:
    for text in (
        "I sold five shirts for 900 birr each.",
        "I sold five shirts for 900 birr each one.",
        "I sold five shirts for 900 birr apiece.",
        "I sold five shirts for 900 birr a piece.",
        "I sold five shirts for 900 birr per shirt.",
    ):
        payload = _data(text)
        assert payload["type"] == "create_event"
        assert payload["data"]["amount"] == 4500
        assert payload["data"]["quantity"] == 5


def test_as_per_is_not_a_unit_price() -> None:
    sale = _data("I sold five shirts for 900 birr as per the receipt.")
    assert sale["type"] == "clarification"
    assert sale["missing_fields"] == ["amount_scope"]
    assert "data" not in sale
    assert "4500" not in str(sale)

    purchase = _data("I bought 20 shirts for 250 birr as per the invoice.")
    assert purchase["type"] == "clarification"
    assert purchase["missing_fields"] == ["amount_scope"]
    assert "5000" not in str(purchase)

    agreement = _data("I sold five shirts for 900 birr as per our agreement.")
    assert agreement["type"] == "clarification"
    assert agreement["missing_fields"] == ["amount_scope"]


def test_total_and_unit_cues_together_ask_instead_of_guessing() -> None:
    payload = _data("I sold five shirts for 900 birr total each.")
    assert payload["type"] == "clarification"
    assert "amount_scope" in payload["missing_fields"]
    assert "data" not in payload


def test_quantity_cannot_become_an_amount() -> None:
    for amount in (Decimal(5), Decimal(500)):
        draft = Draft(
            kind="create_event",
            event_type="sale",
            item="coffees",
            quantity=Decimal(5),
            amount=amount,
            currency="ETB",
            amount_scope="total",
        )
        payload = finalize(draft, "I sold 5 coffees.").model_dump(mode="json", exclude_none=True)
        assert payload["type"] == "clarification"
        assert payload["missing_fields"] == ["amount"]
        assert "data" not in payload

    took = Draft(
        kind="create_event",
        event_type="sale",
        item="coffees",
        quantity=Decimal(5),
        amount=Decimal(5),
        currency="ETB",
        amount_scope="total",
    )
    payload = finalize(took, "Client took 5 coffees.").model_dump(mode="json", exclude_none=True)
    assert payload["type"] == "clarification"
    assert "data" not in payload
    assert "amount" in payload["missing_fields"]


def test_explicit_coffee_prices_keep_their_role() -> None:
    total = _data("I sold 5 coffees for 500 birr total.")
    assert total["data"]["amount"] == 500
    assert total["data"]["quantity"] == 5

    unit = _data("I sold 5 coffees at 100 birr each.")
    assert unit["data"]["amount"] == 500
    assert unit["data"]["quantity"] == 5

    bare = _data("I sold 5 coffees for 500 birr.")
    assert bare["type"] == "clarification"
    assert bare["missing_fields"] == ["amount_scope"]
    assert "data" not in bare


def test_clear_inventory_losses_are_negative() -> None:
    assert _data("I lost 3 shirts.")["data"]["quantity"] == -3
    assert _data("I damaged 2 phones.")["data"] == {
        "item": "phones",
        "quantity": -2,
        "reason": "damaged",
    }
    assert _data("I spoiled 3 shirts.")["data"] == {
        "item": "shirts",
        "quantity": -3,
        "reason": "spoiled",
    }
    assert _data("I removed 3 shirts.")["data"] == {
        "item": "shirts",
        "quantity": -3,
        "reason": "removed",
    }
    assert _data("I broke 4 items.")["data"] == {
        "item": "items",
        "quantity": -4,
        "reason": "broken",
    }
    assert _data("I had 3 shirts stolen.")["data"] == {
        "item": "shirts",
        "quantity": -3,
        "reason": "stolen",
    }


def test_model_cannot_make_a_loss_positive() -> None:
    evidence = analyze("I spoiled 3 shirts.")
    draft = Draft(
        kind="create_event",
        event_type="inventory_adjustment",
        item="shirts",
        quantity=Decimal(3),
        reason="added",
    )
    payload = decide(draft, evidence).model_dump(mode="json", exclude_none=True)
    assert payload["data"] == {"item": "shirts", "quantity": -3, "reason": "spoiled"}

    flipped = finalize(draft, "I removed 3 shirts.").model_dump(mode="json", exclude_none=True)
    assert flipped["data"]["quantity"] == -3
    assert flipped["data"]["reason"] == "removed"


def test_ambiguous_inventory_direction_is_not_trusted() -> None:
    draft = Draft(
        kind="create_event",
        event_type="inventory_adjustment",
        item="shirts",
        quantity=Decimal(3),
        reason="adjusted",
    )
    payload = finalize(draft, "I counted 3 shirts.").model_dump(mode="json", exclude_none=True)
    assert payload["type"] == "clarification"
    assert "data" not in payload

    mixed = _data("I lost 3 shirts and found 3 shirts.")
    assert mixed["type"] == "clarification"
    assert "data" not in mixed


@pytest.mark.parametrize(
    "text",
    [
        "I sold five shirts for 900 birr total yesterday.",
        "I sold five shirts for 900 birr total tomorrow.",
        "I sold five shirts for 900 birr total last week.",
        "I sold five shirts for 900 birr total last month.",
        "I sold five shirts for 900 birr total last year.",
        "I sold five shirts for 900 birr total two days ago.",
        "I sold five shirts for 900 birr total three weeks ago.",
    ],
)
def test_relative_dates_are_not_dropped(text: str) -> None:
    payload = _data(text)
    assert payload["type"] == "clarification"
    assert "date" in payload["missing_fields"]
    assert "data" not in payload
    assert date.today().isoformat() not in str(payload)
    assert "YYYY-MM-DD" in payload["question"]


def test_original_utterance_plus_scope_answer() -> None:
    each = _data("I sold five shirts for 900 birr. Each.")
    assert each["type"] == "create_event"
    assert each["data"]["amount"] == 4500

    total = _data("I sold five shirts for 900 birr. Total.")
    assert total["type"] == "create_event"
    assert total["data"]["amount"] == 900


def test_each_alone_does_not_create_an_event() -> None:
    class Greedy:
        def interpret(self, text: str, language: str) -> Draft:
            return Draft(
                kind="create_event",
                event_type="sale",
                item="shirts",
                quantity=Decimal(5),
                amount=Decimal(900),
                currency="ETB",
                amount_scope="total",
            )

    payload = _data("Each.", Greedy())
    assert payload["type"] == "clarification"
    assert "data" not in payload


def test_replayed_scope_question_uses_the_trailing_answer() -> None:
    class Greedy:
        def interpret(self, text: str, language: str) -> Draft:
            return Draft(
                kind="create_event",
                event_type="sale",
                item="shirts",
                quantity=Decimal(5),
                amount=Decimal(900),
                currency="ETB",
                amount_scope="total",
            )

    question = "Is 900 birr the total for all five shirts, or 900 birr per shirt?"
    each = _data(f"{question} Each.", Greedy())
    assert each["type"] == "create_event"
    assert each["event_type"] == "sale"
    assert each["data"]["amount"] == 4500
    assert each["data"]["quantity"] == 5

    total = _data(f"{question} Total.", Greedy())
    assert total["type"] == "create_event"
    assert total["data"]["amount"] == 900


def _frontend_continuation(original: str, question: str, missing: list[str], answer: str) -> str:
    fields = ", ".join(missing)
    return "\n".join(
        [
            f"Original user statement: {original}",
            f"Clarification question: {question}",
            f"Missing fields: {fields}",
            f"User clarification answer: {answer}",
            "Resolve the original request using the clarification answer.",
        ]
    )


def test_explicit_unit_price_is_the_total_sale() -> None:
    each = _data("I sold 5 shirts for 1000 birr each")
    per_shirt = _data("I sold 5 shirts for 1000 birr per shirt")
    just = _data("I just sold 5 shirts for 1000 birr each")
    for payload in (each, per_shirt, just):
        assert payload["type"] == "create_event"
        assert payload["event_type"] == "sale"
        assert payload["data"] == {
            "item": "shirts",
            "quantity": 5,
            "amount": 5000,
            "currency": "ETB",
        }
        assert "unit_price" not in payload["data"]


def test_explicit_unit_price_purchase_is_the_total() -> None:
    payload = _data("I bought 5 shirts for 1000 birr each")
    assert payload == {
        "type": "create_event",
        "event_type": "purchase",
        "data": {"item": "shirts", "quantity": 5, "amount": 5000, "currency": "ETB"},
    }


def test_unqualified_price_with_quantity_stays_ambiguous() -> None:
    payload = _data("I sold five shirts for 1000 birr")
    assert payload["type"] == "clarification"
    assert payload["missing_fields"] == ["amount_scope"]
    assert "1000 birr" in payload["question"]
    assert "per shirt" in payload["question"]


def test_amount_answer_continues_the_original_sale() -> None:
    class Boom:
        def interpret(self, text: str, language: str) -> Draft:
            raise AssertionError(text)

    payload = _data(
        _frontend_continuation(
            "I sold five shirts",
            "How much was the sale?",
            ["amount"],
            "1000 birr",
        ),
        Boom(),
    )
    assert payload == {
        "type": "create_event",
        "event_type": "sale",
        "data": {"item": "shirts", "quantity": 5, "amount": 1000, "currency": "ETB"},
    }


def test_scope_answer_resolves_an_ambiguous_sale() -> None:
    per_shirt = _data(
        _frontend_continuation(
            "I sold five shirts for 1000 birr",
            "Is 1000 birr the total for all five shirts, or 1000 birr per shirt?",
            ["amount_scope"],
            "per shirt",
        )
    )
    assert per_shirt["type"] == "create_event"
    assert per_shirt["event_type"] == "sale"
    assert per_shirt["data"]["quantity"] == 5
    assert per_shirt["data"]["amount"] == 5000
    assert per_shirt["data"]["currency"] == "ETB"

    total = _data(
        _frontend_continuation(
            "I sold five shirts for 1000 birr",
            "Is 1000 birr the total for all five shirts, or 1000 birr per shirt?",
            ["amount_scope"],
            "total",
        )
    )
    assert total["type"] == "create_event"
    assert total["data"]["amount"] == 1000
    assert total["data"]["quantity"] == 5


def test_scope_answer_does_not_invent_a_quantity() -> None:
    per_shirt = _data(
        _frontend_continuation(
            "I sold shirts for 1000 birr",
            "Is 1000 birr the total for all the shirts, or 1000 birr per shirt?",
            ["quantity", "amount_scope"],
            "per shirt",
        )
    )
    assert per_shirt["type"] == "clarification"
    assert per_shirt["missing_fields"] == ["quantity"]
    assert "amount" not in per_shirt["missing_fields"]
    assert "data" not in per_shirt

    total = _data(
        _frontend_continuation(
            "I sold shirts for 1000 birr",
            "Is 1000 birr the total for all the shirts, or 1000 birr per shirt?",
            ["quantity"],
            "total",
        )
    )
    assert total["type"] == "clarification"
    assert total["missing_fields"] == ["quantity"]
    assert "How many shirts did you sell?" in total["question"]


def test_wrapped_new_sale_is_not_folded_into_an_expense() -> None:
    payload = _data(
        _frontend_continuation(
            "I spent 200 birr on rent",
            "What was the expense for?",
            ["description"],
            "I just sold 5 shirts for 1000 birr each",
        )
    )
    assert payload["type"] == "create_event"
    assert payload["event_type"] == "sale"
    assert payload["data"]["amount"] == 5000
    assert payload["data"]["quantity"] == 5
