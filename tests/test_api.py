"""HTTP contract for health and interpret."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.interpreter.raw import Draft, LLMRaw
from app.llm.client import GeminiChatClient
from app.llm.errors import ProviderBadResponse, ProviderUnavailable
from app.llm.provider import GeminiProvider
from app.main import create_app


def test_health_does_not_call_the_provider() -> None:
    class Boom:
        def interpret(self, text: str, language: str) -> Draft:
            raise AssertionError("health called the provider")

    client = TestClient(create_app(Boom()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_interpret_sale_over_http() -> None:
    client = TestClient(create_app(provider=None))
    response = client.post(
        "/interpret",
        json={"text": "I sold five shirts for 900 birr each.", "language": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["amount"] == 4500
    assert "customer" not in body["data"]
    assert "confirmed" not in body
    assert "recorded" not in body


def test_interpret_ambiguous_sale_over_http() -> None:
    client = TestClient(create_app(provider=None))
    response = client.post(
        "/interpret",
        json={"text": "I sold five shirts for 900 birr.", "language": "en"},
    )
    assert response.status_code == 200
    assert response.json()["type"] == "clarification"
    assert response.json()["missing_fields"] == ["amount_scope"]


def test_language_defaults_and_business_id_is_not_required() -> None:
    client = TestClient(create_app(provider=None))
    response = client.post(
        "/interpret",
        json={"text": "How much did I sell today?", "business_id": "ignored"},
    )
    assert response.status_code == 200
    assert response.json()["type"] == "query"


def test_malformed_request_is_422() -> None:
    client = TestClient(create_app(provider=None))
    assert client.post("/interpret", json={}).status_code == 422
    assert client.post("/interpret", json={"text": ""}).status_code == 422
    assert client.post("/interpret", json={"text": "   "}).status_code == 422
    assert client.post("/interpret", json={"text": "hi", "language": "1"}).status_code == 422


def test_provider_unavailable_is_503() -> None:
    class Down:
        def interpret(self, text: str, language: str) -> Draft:
            raise ProviderUnavailable()

    client = TestClient(create_app(Down()))
    response = client.post("/interpret", json={"text": "Client took coffees home."})
    assert response.status_code == 503
    assert response.json() == {"detail": "Interpretation provider is unavailable."}
    assert "key" not in response.text.lower()


def test_provider_bad_response_is_502() -> None:
    class Bad:
        def interpret(self, text: str, language: str) -> Draft:
            raise ProviderBadResponse()

    client = TestClient(create_app(Bad()))
    response = client.post("/interpret", json={"text": "Client took coffees home."})
    assert response.status_code == 502
    assert "secret" not in response.text.lower()


def test_unexpected_error_is_500_without_the_exception_text() -> None:
    class Boom:
        def interpret(self, text: str, language: str) -> Draft:
            raise RuntimeError("sk-live-secret-do-not-leak")

    client = TestClient(create_app(Boom()))
    response = client.post("/interpret", json={"text": "Client took coffees home."})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "sk-live-secret" not in response.text


def test_provider_retries_malformed_output_once() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, str]]) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                return {"kind": "create_event", "quantity": "nope"}
            return LLMRaw(kind="query", query="ignored").model_dump()

    fake = FakeClient()
    provider = GeminiProvider(api_key="test-key", client=fake)
    draft = provider.interpret("What did I sell today?", "en")
    assert fake.calls == 2
    assert draft.kind == "query"


def test_provider_gives_up_after_two_bad_payloads() -> None:
    class FakeClient:
        def complete(self, messages: list[dict[str, str]]) -> dict[str, object]:
            return {"kind": "not-a-kind"}

    provider = GeminiProvider(api_key="test-key", client=FakeClient())
    with pytest.raises(ProviderBadResponse):
        provider.interpret("hello", "en")


def test_blank_gemini_key_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="  ")


def test_gemini_client_rejects_malformed_json_without_calling_out() -> None:
    class Offline(GeminiChatClient):
        def _generate(self, messages: list[dict[str, str]]) -> str:
            return "not json"

    with pytest.raises(ProviderBadResponse):
        Offline(api_key="test-key", model="gemini-3.1-flash-lite", timeout_seconds=1).complete(
            [{"role": "user", "content": "hello"}]
        )


def test_gemini_client_parses_structured_json() -> None:
    class Offline(GeminiChatClient):
        def _generate(self, messages: list[dict[str, str]]) -> str:
            return '{"kind":"clarification","query":null}'

    raw = Offline(api_key="test-key", model="gemini-3.1-flash-lite", timeout_seconds=1).complete(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
        ]
    )
    assert raw.kind == "clarification"


def test_interpret_options_preflight_allows_localhost() -> None:
    client = TestClient(create_app(provider=None))
    response = client.options(
        "/interpret",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in response.headers.get("access-control-allow-methods", "").upper()
    assert "content-type" in response.headers.get("access-control-allow-headers", "").lower()


def test_interpret_options_preflight_allows_production_frontend() -> None:
    client = TestClient(create_app(provider=None))
    origin = "https://voice-first-business-assistant.vercel.app"
    response = client.options(
        "/interpret",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_interpret_post_includes_cors_origin_header() -> None:
    client = TestClient(create_app(provider=None))
    response = client.post(
        "/interpret",
        headers={"Origin": "http://localhost:3000"},
        json={"text": "I sold five shirts for 900 birr each.", "language": "en"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.json()["data"]["amount"] == 4500


def test_cors_origins_env_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "http://localhost:3000,https://voice-first-business-assistant.vercel.app",
    )
    client = TestClient(create_app(provider=None))
    response = client.options(
        "/interpret",
        headers={
            "Origin": "https://voice-first-business-assistant.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "https://voice-first-business-assistant.vercel.app"
    )
