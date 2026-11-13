"""
tests/test_gemini.py
=====================
Tests for the Gemini LLM integration (Phase 2).

Two test groups are provided:

1. **Unit tests (mocked)**
   Run without network access or a real API key.  They verify that the
   ``get_llm()`` factory behaves correctly (caching, error handling) by
   patching the underlying ``ChatGoogleGenerativeAI`` class.

   These tests prove that the wiring is correct, **not** that the real
   Gemini API works.

2. **Integration test (real API)**
   Sends a live prompt to Google Gemini and validates the response.
   Requires ``GOOGLE_API_KEY`` to be set in the environment (or ``.env``).
   Automatically **skipped** when no valid key is available.

Run all tests::

    python -m pytest tests/test_gemini.py -v

Run only the mocked (offline) tests::

    python -m pytest tests/test_gemini.py -v -k "not integration"

Run only the integration test (requires a real key)::

    python -m pytest tests/test_gemini.py -v -k "integration"
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Load .env so GOOGLE_API_KEY is available when running via pytest.
load_dotenv(override=False)


def _has_real_google_api_key() -> bool:
    """Return True if a non-placeholder GOOGLE_API_KEY is available."""
    key = os.environ.get("GOOGLE_API_KEY", "")
    placeholders = {"", "your_google_api_key", "your_google_api_key_here"}
    return bool(key) and key.strip().lower() not in placeholders


def extract_text(content: str | list | None) -> str:
    """Extract plain text from a Gemini response's ``content`` field.

    ``langchain-google-genai`` may return ``content`` as:

    * A plain **string** (older models / older library versions).
    * A **list** of content blocks (Gemini 3+ models), where each block
      is typically a dict like ``{"type": "text", "text": "..."}`` or
      ``{"type": "thinking", "thinking": "..."}``, or occasionally a
      plain string.

    This helper normalises both formats into a single concatenated
    string, extracting only user-facing text (``"type": "text"``
    blocks).  Thinking blocks and non-text blocks are silently skipped.

    Parameters
    ----------
    content:
        The ``response.content`` value from an ``AIMessage``.

    Returns
    -------
    str
        The combined text content, or an empty string if nothing
        could be extracted.
    """
    if content is None:
        return ""

    # Simple string content – return as-is.
    if isinstance(content, str):
        return content

    # List of content blocks.
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block:
                    parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                block_text = block.get("text")
                if block_text is None:
                    continue
                # Standard text block: {"type": "text", "text": "..."}
                if block_type == "text" or block_type not in (
                    "thinking",
                    "executable_code",
                    "code_execution_result",
                ):
                    parts.append(str(block_text))
            # Skip anything else (MagicMock, unexpected types, etc.)
        return "\n".join(parts)

    if isinstance(content, dict):
        return extract_text([content])

    # Unexpected type – handle gracefully.
    return ""


# ---------------------------------------------------------------------------
# Unit Tests – Mocked (no API key required)
# ---------------------------------------------------------------------------


class TestGetLLMMocked:
    """Unit tests for ``get_llm()`` using a mocked ChatGoogleGenerativeAI."""

    @pytest.fixture(autouse=True)
    def _reset_caches(self):
        """Clear the ``get_llm`` and ``get_settings`` caches between tests."""
        from backend.config import get_settings
        from backend.llm import get_llm

        get_settings.cache_clear()
        get_llm.cache_clear()
        yield
        get_settings.cache_clear()
        get_llm.cache_clear()

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "mock-key-for-testing",
            "TAVILY_API_KEY": "mock-tavily-key",
            "GEMINI_MODEL": "gemini-2.5-flash",
        },
    )
    @patch("backend.llm.ChatGoogleGenerativeAI")
    def test_get_llm_returns_model_instance(self, mock_chat_cls: MagicMock) -> None:
        """``get_llm()`` should return the object created by ChatGoogleGenerativeAI."""
        from backend.llm import get_llm

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        llm = get_llm()

        assert llm is mock_instance
        mock_chat_cls.assert_called_once()

        # Verify correct kwargs were passed.
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 8192

        print("\n✅ [MOCKED] get_llm() returned a valid model instance.")

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "mock-key-for-testing",
            "TAVILY_API_KEY": "mock-tavily-key",
            "GEMINI_MODEL": "gemini-3.8-flash",
        },
    )
    @patch("backend.llm.ChatGoogleGenerativeAI")
    def test_get_llm_is_cached(self, mock_chat_cls: MagicMock) -> None:
        """Repeated calls to ``get_llm()`` should return the same instance."""
        from backend.llm import get_llm

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        llm1 = get_llm()
        llm2 = get_llm()

        assert llm1 is llm2
        # Constructor should only be called once.
        assert mock_chat_cls.call_count == 1

        print("\n✅ [MOCKED] get_llm() correctly caches the model instance.")

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "mock-key-for-testing",
            "TAVILY_API_KEY": "mock-tavily-key",
            "GEMINI_MODEL": "gemini-3.8-flash",
        },
    )
    @patch("backend.llm.ChatGoogleGenerativeAI")
    def test_get_llm_uses_configured_model(self, mock_chat_cls: MagicMock) -> None:
        """``get_llm()`` should use the GEMINI_MODEL from the configuration."""
        from backend.llm import get_llm

        mock_chat_cls.return_value = MagicMock()

        get_llm()

        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.8-flash"

        print("\n✅ [MOCKED] get_llm() used the configured model name.")

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "mock-key-for-testing",
            "TAVILY_API_KEY": "mock-tavily-key",
            "GEMINI_MODEL": "gemini-3.8-flash",
        },
    )
    @patch("backend.llm.ChatGoogleGenerativeAI", side_effect=Exception("init failed"))
    def test_get_llm_raises_on_init_failure(self, mock_chat_cls: MagicMock) -> None:
        """``get_llm()`` should raise RuntimeError when the model fails to init."""
        from backend.llm import get_llm

        with pytest.raises(RuntimeError, match="Failed to initialise Gemini model"):
            get_llm()

        print("\n✅ [MOCKED] get_llm() raised RuntimeError on init failure.")

    @patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "mock-key-for-testing",
            "TAVILY_API_KEY": "mock-tavily-key",
            "GEMINI_MODEL": "gemini-3.8-flash",
        },
    )
    @patch("backend.llm.ChatGoogleGenerativeAI")
    def test_mocked_invoke(self, mock_chat_cls: MagicMock) -> None:
        """Mocked invoke should return the stubbed response content."""
        from backend.llm import get_llm

        mock_response = MagicMock()
        mock_response.content = "Mocked: Visit Munnar, Alleppey, and Wayanad."
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_instance

        llm = get_llm()
        response = llm.invoke(
            "You are an AI travel assistant. Suggest three interesting "
            "destinations in Kerala, India, and briefly explain why each "
            "is worth visiting."
        )

        assert response.content
        assert len(response.content) > 0
        print(f"\n✅ [MOCKED] invoke returned: {response.content}")
        print(
            "\n⚠️  Note: This mocked test verifies wiring only. "
            "It does NOT prove the real Gemini API works."
        )


# ---------------------------------------------------------------------------
# Integration Test – Real API Call
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_real_google_api_key(),
    reason=(
        "GOOGLE_API_KEY is not set or contains a placeholder. "
        "Set a valid key in .env to run integration tests."
    ),
)
class TestGeminiIntegration:
    """Integration tests that call the real Gemini API."""

    @pytest.fixture(autouse=True)
    def _reset_caches(self):
        """Clear caches so a fresh Settings/LLM is constructed."""
        from backend.config import get_settings
        from backend.llm import get_llm

        get_settings.cache_clear()
        get_llm.cache_clear()
        yield
        get_settings.cache_clear()
        get_llm.cache_clear()

    def test_integration_gemini_response(self) -> None:
        """Send a real prompt to Gemini and validate the response.

        This test requires:
        - A valid ``GOOGLE_API_KEY`` in the environment or ``.env``.
        - Network connectivity to the Google Generative AI API.
        """
        from backend.llm import get_llm

        llm = get_llm()

        prompt = (
            "You are an AI travel assistant. Suggest three interesting "
            "destinations in Kerala, India, and briefly explain why each "
            "is worth visiting."
        )

        try:
            response = llm.invoke(prompt)
        except Exception as exc:
            error_msg = str(exc).lower()
            if "api key" in error_msg or "permission" in error_msg:
                pytest.fail(
                    f"Gemini API authentication failed. "
                    f"Check that GOOGLE_API_KEY is valid: {exc}"
                )
            elif "quota" in error_msg or "rate" in error_msg:
                pytest.skip(f"Gemini API quota exceeded: {exc}")
            elif "connect" in error_msg or "network" in error_msg:
                pytest.skip(f"Network connectivity issue: {exc}")
            else:
                pytest.fail(f"Gemini API call failed unexpectedly: {exc}")

        # ── Validate the response ────────────────────────────────────────
        assert response is not None, "Response object is None."
        assert hasattr(response, "content"), "Response has no 'content' attribute."
        assert response.content, "Response content is empty."

        text = extract_text(response.content)

        assert text, (
            "Could not extract any text from response.content. "
            f"Raw type: {type(response.content).__name__}"
        )
        assert len(text.strip()) > 50, (
            f"Response is too short to be meaningful "
            f"({len(text.strip())} chars)."
        )

        print("\n" + "=" * 60)
        print("  ✅ GEMINI INTEGRATION TEST – SUCCESS")
        print("=" * 60)
        print(f"\n📝 Prompt:\n  {prompt}")
        print(f"\n📦 Content type: {type(response.content).__name__}")
        print(f"\n🤖 Response ({len(text)} chars):\n")
        print(text)
        print("\n" + "=" * 60)

