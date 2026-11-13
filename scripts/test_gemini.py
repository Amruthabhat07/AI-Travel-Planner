"""
scripts/test_gemini.py
=======================
Interactive test script for the Gemini LLM integration.

Initialises the Gemini model via ``get_llm()`` and accepts a user prompt
from the terminal.  The response is streamed back to the console.

Usage::

    python -m scripts.test_gemini

Example interaction::

    Enter your travel question (or 'quit' to exit):
    Suggest a 3-day trip to Coorg.

    AI Response:
    [Gemini-generated response]
"""

from __future__ import annotations

import sys
import textwrap


def main() -> int:
    """Run the interactive Gemini test loop.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    print("\n" + "=" * 60)
    print("  AI Travel Planner – Gemini LLM Interactive Test")
    print("=" * 60)

    # ── Load configuration ────────────────────────────────────────────────
    try:
        from backend.config import get_settings

        get_settings.cache_clear()
        cfg = get_settings()
    except Exception as exc:
        print(f"\n❌ Configuration error: {exc}")
        print(
            "\n💡 Make sure your .env file exists and contains valid API keys."
        )
        print("   Copy .env.example to .env and fill in your credentials.")
        return 1

    # ── Initialise LLM ────────────────────────────────────────────────────
    try:
        from backend.llm import get_llm

        get_llm.cache_clear()
        llm = get_llm()
    except ValueError as exc:
        print(f"\n❌ Missing API key: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"\n❌ LLM initialisation failed: {exc}")
        return 1
    except Exception as exc:
        print(f"\n❌ Unexpected error during LLM init: {exc}")
        return 1

    print(f"\n✅ Gemini model initialised: {cfg.gemini_model}")
    print("   Type 'quit' or 'exit' to stop.\n")

    # ── Interactive loop ──────────────────────────────────────────────────
    while True:
        try:
            prompt = input("Enter your travel question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            return 0

        if not prompt:
            print("⚠️  Empty prompt. Please enter a question.\n")
            continue

        if prompt.lower() in {"quit", "exit", "q"}:
            print("\nGoodbye!")
            return 0

        print("\n⏳ Generating response…\n")

        try:
            response = llm.invoke(prompt)
        except Exception as exc:
            error_msg = str(exc).lower()
            if "api key" in error_msg or "permission" in error_msg:
                print(f"❌ Authentication error: {exc}")
                print("   Check that your GOOGLE_API_KEY is valid.\n")
            elif "quota" in error_msg or "rate" in error_msg:
                print(f"❌ Quota / rate limit exceeded: {exc}")
                print("   Wait a moment and try again.\n")
            elif "connect" in error_msg or "network" in error_msg:
                print(f"❌ Network error: {exc}")
                print("   Check your internet connection.\n")
            elif "not found" in error_msg or "not supported" in error_msg:
                print(f"❌ Unsupported model: {exc}")
                print(
                    "   Update GEMINI_MODEL in .env to a supported model ID.\n"
                )
            else:
                print(f"❌ API error: {exc}\n")
            continue

        # ── Display response ──────────────────────────────────────────────
        if not response or not getattr(response, "content", None):
            print("⚠️  The model returned an empty response. Try rephrasing.\n")
            continue

        print("AI Response:")
        print("-" * 40)
        # Wrap long lines for terminal readability.
        for line in response.content.splitlines():
            print(textwrap.fill(line, width=80))
        print("-" * 40)
        print()

    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())

