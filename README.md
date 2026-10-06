# Meri AI Engine

Interpretation service for [Meri](https://github.com/Fraol-D) (መሪ), a voice-first assistant for small businesses.

The Meri backend is the source of truth. This service only reads one utterance and returns a typed interpretation. It does **not** persist business data, call the Meri backend, call Voxide, calculate business metrics, or record events.

The client decides whether to ask a follow-up question, ask the user to confirm, call `/events`, or call `/query`.

## What it returns

`POST /interpret` returns one of three results.

### Event

Enough information for a backend event. The service does not confirm or record it.

```json
{
  "type": "create_event",
  "event_type": "sale",
  "data": {
    "item": "shirts",
    "quantity": 5,
    "amount": 4500,
    "currency": "ETB"
  }
}
```

`event_type` is `sale`, `expense`, `purchase`, `inventory_adjustment`, or `customer_debt`. `data` uses the fields the Meri `POST /api/v1/events` contract already accepts.

### Query

The user's question is preserved. The answer is not calculated.

```json
{
  "type": "query",
  "query": "How much did I sell today?"
}
```

### Clarification

A required or ambiguous field is missing. The question says what to add. Vague words such as "some" or "a few" are never turned into numbers.

```json
{
  "type": "clarification",
  "question": "Is 900 birr the total for all five shirts, or 900 birr per shirt?",
  "missing_fields": ["amount_scope"]
}
```

Price scope is not guessed. "900 birr" with a quantity is a clarification. "900 birr total" is the transaction amount. "900 birr each" is a unit price, and the only multiplication this service does is `quantity × unit price`.

`birr`, `ETB`, and `Ethiopian birr` become `ETB`. A currency the user did not name is left out. A different currency the user did name is kept. Relative dates such as "yesterday" are not converted to today.

## Local setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

## Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | No | empty | Gemini API key, used only on the server. Recognized English patterns work without it. |
| `GEMINI_MODEL` | No | `gemini-3.1-flash-lite` | Model id for unrecognized phrasing. This default supports structured JSON on the free tier. |
| `GEMINI_TIMEOUT_SECONDS` | No | `30` | Provider timeout. |

The key is read from the environment or from a git-ignored `.env`. It is never returned to the client.

Without `GEMINI_API_KEY`, an utterance the English rules do not recognize returns a clarification. The service does not invent a model response.

## Run

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Render uses `render.yaml`. The service listens on `0.0.0.0` and `$PORT`. Set `GEMINI_API_KEY` in the Render dashboard if unrecognized phrasing should call Gemini. Leave it empty to run the English rules only.

Health check (this does not call the model):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

Interpret:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/interpret -ContentType "application/json" -Body '{"text":"I sold five shirts for 900 birr total.","language":"en"}'
```

`language` defaults to `en`. `business_id` is not required. English is the supported MVP language.

### Status codes

| Situation | Status |
| --- | --- |
| Health, or a valid interpret request | 200 |
| Missing or invalid JSON body | 422 |
| Provider cannot be reached | 503 |
| Provider returns an unusable payload after one retry | 502 |
| Unexpected server error | 500 |

Error bodies are `{"detail": "..."}`. They do not include provider payloads or credentials.

## Provider

One provider is implemented: Gemini via the `google-genai` SDK, behind `LLMProvider`. The default model is `gemini-3.1-flash-lite`. The model is asked for the `LLMRaw` JSON schema. That JSON is validated with Pydantic, then checked against the source text.

The model is not the only safety check. After any model output, the service rejects unsupported event types, non-positive amounts, zero inventory changes, invented numbers, vague quantities, and an unresolved price scope.

Recognized English sales, purchases, expenses, debts, inventory notes, and questions are interpreted by deterministic rules so those cases do not depend on a live model call.

## Tests

```powershell
pytest
ruff check .
```

## Architecture

```text
utterance
  -> English rules, when the speech act is recognized
  -> otherwise Gemini structured output
  -> Pydantic validation
  -> semantic checks and normalization
  -> InterpretationResult
```

```text
app/
  main.py                  # app factory
  config.py                # environment
  api/routes.py            # GET /health, POST /interpret
  schemas/                 # public request and response models
  interpreter/             # rules, prompts, normalization, validation
  llm/                     # provider interface and Gemini client
tests/
```

This process has no database, queue, cache, ORM, MCP server, LangChain agent, or HTTP client for the Meri backend or Voxide.

## Limitations

This is an MVP interpreter, not a production service.

- There is no authentication, rate limit, or audit log.
- English rules cover the bookkeeping phrases in the test suite. Other wording is sent to Gemini only when `GEMINI_API_KEY` is set, and that live path is not part of the automated tests.
- Amharic and other languages are accepted in the `language` field but are not parsed by the rules. Without a provider key they return a clarification.
- Each request is stateless. There is no conversation store. After a clarification, send the original utterance plus the user's answer, for example `I sold five shirts for 900 birr. Each.` `Each.` or `Total.` alone does not create an event. The generated clarification question does not have to be sent. If it is included and the text ends in a short `Each.` or `Total.`, that answer is the amount-scope cue. The original sentence still has to be in the text when no model is configured, because the question itself does not say sold or bought.
- Relative dates such as yesterday, last week, last year, and two days ago are not calculated and are not dropped. The response asks for a concrete `YYYY-MM-DD` date instead of guessing or using today.
- The service never says an event was recorded.
