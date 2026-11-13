"""
backend/config.py
==================
Centralised configuration management for the AI Travel Planner.

Loads all environment variables from a .env file using python-dotenv and
exposes them as a validated, strongly-typed Settings object via
pydantic-settings. Required keys raise a descriptive error on startup if
they are missing or empty; optional keys fall back to safe defaults.

Usage::

    from backend.config import settings

    api_key = settings.google_api_key
    is_tracing_enabled = settings.langsmith_tracing
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Locate and load the .env file from the project root (two levels up from
# this file: backend/config.py → backend/ → project root).
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_ENV_FILE: Path = _PROJECT_ROOT / ".env"

# Load early so that environment variables are available before Settings is
# instantiated.  load_dotenv() is a no-op when the file does not exist.
load_dotenv(dotenv_path=_ENV_FILE, override=False)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings Model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables.

    Required variables raise a ``ValueError`` at startup if absent;
    optional variables fall back to their declared defaults.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow extra fields to avoid errors if .env has future keys.
        extra="ignore",
    )

    # ── Required ──────────────────────────────────────────────────────────────

    google_api_key: str = Field(
        ...,
        description="Google Gemini API key (https://aistudio.google.com/app/apikey).",
    )

    tavily_api_key: str = Field(
        ...,
        description="Tavily web search API key (https://tavily.com/).",
    )

    # ── Optional: Gemini Model ────────────────────────────────────────────────

    gemini_model: str = Field(
        default="gemini-3.8-flash",
        description=(
            "Gemini model ID to use for LLM calls. "
            "Must be a model supported by the Google Generative AI API."
        ),
    )

    # ── Optional: LangSmith ───────────────────────────────────────────────────

    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key for agent tracing (optional).",
    )

    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith request tracing when True.",
    )

    langsmith_project: str = Field(
        default="ai-travel-planner",
        description="LangSmith project name.",
    )

    # ── Optional: Future integrations ────────────────────────────────────────

    geoapify_api_key: Optional[str] = Field(
        default=None,
        description="Geoapify API key for location and places (future).",
    )

    # ── Application ───────────────────────────────────────────────────────────

    app_env: str = Field(
        default="development",
        description="Deployment environment: development | staging | production.",
    )

    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ── FastAPI ───────────────────────────────────────────────────────────────

    api_host: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    api_port: int = Field(default=8000, description="Uvicorn bind port.")

    # ── Database ──────────────────────────────────────────────────────────────

    database_url: str = Field(
        default="sqlite:///./travel_planner.db",
        description="SQLAlchemy database connection URL.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("google_api_key", "tavily_api_key", mode="before")
    @classmethod
    def _reject_placeholder_values(cls, value: str, info) -> str:  # type: ignore[override]
        """
        Guard against accidentally shipping placeholder strings from
        .env.example into the runtime environment.
        """
        placeholders = {
            "your_google_api_key_here",
            "your_tavily_api_key_here",
            "",
        }
        if not value or value.strip().lower() in placeholders:
            field_name = info.field_name if info else "unknown"
            raise ValueError(
                f"\n\n[AI Travel Planner – Configuration Error]\n"
                f"  Required environment variable '{field_name.upper()}' is missing or "
                f"contains a placeholder value.\n"
                f"  → Copy .env.example to .env and set a real API key.\n"
                f"  → Obtain a Google Gemini key at: https://aistudio.google.com/app/apikey\n"
                f"  → Obtain a Tavily key at: https://tavily.com/\n"
            )
        return value

    @model_validator(mode="after")
    def _warn_langsmith_misconfiguration(self) -> "Settings":
        """
        Emit a warning if tracing is enabled but no API key is provided.
        This is non-fatal so that developers can run locally without tracing.
        """
        if self.langsmith_tracing and not self.langsmith_api_key:
            logger.warning(
                "[AI Travel Planner] LANGSMITH_TRACING is True but "
                "LANGSMITH_API_KEY is not set. Tracing will be disabled."
            )
            # Coerce tracing off to prevent LangSmith SDK errors at runtime.
            object.__setattr__(self, "langsmith_tracing", False)
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        """Return True when running in the production environment."""
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Return True when running in the development environment."""
        return self.app_env.lower() == "development"

    def configure_langsmith(self) -> None:
        """
        Propagate LangSmith settings to the environment so that the
        LangSmith SDK picks them up automatically.
        Call this once at application startup.
        """
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            logger.info(
                "[AI Travel Planner] LangSmith tracing enabled "
                "(project: %s).",
                self.langsmith_project,
            )
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ---------------------------------------------------------------------------
# Singleton accessor (cached so Settings is only constructed once per process)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application-wide :class:`Settings` singleton.

    This function is cached: the Settings object is constructed only on the
    first call and reused on every subsequent call.  Use
    ``get_settings.cache_clear()`` in tests to reset between test cases.

    Raises:
        pydantic.ValidationError: If a required environment variable is
            missing or contains a placeholder value.
    """
    return Settings()  # type: ignore[call-arg]


# Convenience alias for direct import::
#
#     from backend.config import settings
#     print(settings.google_api_key)
#
# Note: this module-level alias is NOT cached itself; it triggers
# ``get_settings()`` which IS cached.  Importing ``settings`` directly is
# safe, but make sure to call ``get_settings.cache_clear()`` between tests
# rather than reimporting ``settings``.
try:
    settings = get_settings()
except Exception:  # noqa: BLE001
    # Allow the module to be imported without a valid .env so that
    # verify_setup.py and tests can inspect the config class before
    # environment variables are available.
    settings = None  # type: ignore[assignment]
