"""Runtime settings. Provider credentials stay in the server environment."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    """Load a local env file without overriding variables that are already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    timeout_raw = os.getenv("GEMINI_TIMEOUT_SECONDS", "30").strip()
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 30.0
    if timeout <= 0:
        timeout = 30.0
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip() or "gemini-3.1-flash-lite"
    cors_raw = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).strip() or DEFAULT_CORS_ORIGINS
    return Settings(
        gemini_api_key=key or None,
        gemini_model=model,
        gemini_timeout_seconds=timeout,
        cors_origins=parse_cors_origins(cors_raw),
    )


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,"
    "https://voice-first-business-assistant.vercel.app"
)


def parse_cors_origins(raw: str) -> list[str]:
    """Split a comma-separated CORS origin list. Empty entries are dropped."""
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_timeout_seconds: float = Field(default=30.0, gt=0)
    cors_origins: list[str] = Field(
        default_factory=lambda: parse_cors_origins(DEFAULT_CORS_ORIGINS),
    )
