"""Health and interpret routes. Health never calls the provider."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.interpreter.continuation import Continuation
from app.interpreter.service import EmptyUtterance, InterpreterService
from app.llm.errors import ProviderBadResponse, ProviderUnavailable
from app.schemas.requests import InterpretRequest
from app.schemas.responses import InterpretationResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/interpret", response_model=InterpretationResult, response_model_exclude_none=True)
def interpret(body: InterpretRequest, request: Request) -> JSONResponse:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    service: InterpreterService = request.app.state.interpreter
    try:
        result = service.interpret(text, body.language, _continuation(body))
    except EmptyUtterance:
        raise HTTPException(status_code=422, detail="text must not be empty") from None
    except ProviderUnavailable:
        return JSONResponse(
            status_code=503,
            content={"detail": ProviderUnavailable.detail},
        )
    except ProviderBadResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": ProviderBadResponse.detail},
        )
    except Exception as exc:
        logger.error("Unexpected interpretation failure: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
    return JSONResponse(status_code=200, content=result.model_dump(mode="json", exclude_none=True))


def _continuation(body: InterpretRequest) -> Continuation | None:
    context = body.context
    if context is None or not context.original_text.strip():
        return None
    previous = context.previous_result
    missing = tuple(previous.missing_fields) if previous is not None else ()
    # The question is accepted and ignored. Only the original utterance is
    # evidence, so a question that mentions a quantity cannot invent one.
    return Continuation(
        original_text=context.original_text.strip(),
        missing_fields=missing,
    )
