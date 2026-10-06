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
    key = os.getenv("XAI_API_KEY", "").strip()
    timeout_raw = os.getenv("XAI_TIMEOUT_SECONDS", "30").strip()
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 30.0
    if timeout <= 0:
        timeout = 30.0
    model = os.getenv("XAI_MODEL", "grok-4.7").strip() or "grok-4.7"
    return Settings(
        xai_api_key=key or None,
        xai_model=model,
        xai_timeout_seconds=timeout,
    )


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    xai_api_key: str | None = None
    xai_model: str = "grok-4.7"
    xai_timeout_seconds: float = Field(default=30.0, gt=0)
