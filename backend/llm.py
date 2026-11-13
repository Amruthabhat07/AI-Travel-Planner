"""
backend/llm.py
==============
Reusable LLM initialisation module for the AI Travel Planner.

Provides a singleton :class:`~langchain_google_genai.ChatGoogleGenerativeAI`
instance that all specialised agents can share.  The model, temperature, and
output-token limit are driven by environment variables (via
:mod:`backend.config`) so nothing is hardcoded.

Usage::

    from backend.llm import get_llm

    llm = get_llm()
    response = llm.invoke("Suggest a weekend trip from Bangalore.")
    print(response.content)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridable via environment / Settings)
# ---------------------------------------------------------------------------

_DEFAULT_TEMPERATURE: float = 0.7
_DEFAULT_MAX_OUTPUT_TOKENS: int = 8192


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm(
    *,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> ChatGoogleGenerativeAI:
    """Return a cached :class:`ChatGoogleGenerativeAI` instance.

    The instance is created once and reused across the entire application
    process, avoiding unnecessary re-initialisation for every agent
    invocation.

    Parameters
    ----------
    temperature:
        Sampling temperature (0.0 – 2.0).  Lower values produce more
        deterministic output.  Defaults to ``0.7``.
    max_output_tokens:
        Maximum number of tokens in the model response.  Defaults to
        ``8192``.

    Returns
    -------
    ChatGoogleGenerativeAI
        A ready-to-use LangChain chat model bound to Google Gemini.

    Raises
    ------
    ValueError
        If ``GOOGLE_API_KEY`` is missing or contains a placeholder value.
    RuntimeError
        If the model fails to initialise (e.g. unsupported model name,
        invalid API key format).
    """
    cfg = get_settings()

    # ── Validate API key ──────────────────────────────────────────────────
    api_key: str = cfg.google_api_key
    if not api_key or api_key.strip() == "":
        raise ValueError(
            "[AI Travel Planner] GOOGLE_API_KEY is not set. "
            "Add it to your .env file. "
            "Obtain a key at: https://aistudio.google.com/app/apikey"
        )

    model_name: str = cfg.gemini_model
    temp: float = temperature if temperature is not None else _DEFAULT_TEMPERATURE
    max_tokens: int = (
        max_output_tokens if max_output_tokens is not None else _DEFAULT_MAX_OUTPUT_TOKENS
    )

    # ── Initialise the model ──────────────────────────────────────────────
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        # Provide a user-friendly message without leaking the API key.
        raise RuntimeError(
            f"[AI Travel Planner] Failed to initialise Gemini model "
            f"'{model_name}': {exc}\n"
            f"  → Verify that GOOGLE_API_KEY is valid.\n"
            f"  → Verify that GEMINI_MODEL is a supported model ID.\n"
            f"  → See https://ai.google.dev/gemini-api/docs/models"
        ) from exc

    logger.info(
        "[AI Travel Planner] Gemini LLM initialised. "
        "model=%s | temperature=%.2f | max_tokens=%d",
        model_name,
        temp,
        max_tokens,
    )
    return llm

