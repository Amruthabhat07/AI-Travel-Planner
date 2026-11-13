"""Tavily-backed web search utility, independent of agent behavior."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.tools import ToolException
from langchain_tavily import TavilySearch
from langchain_tavily.tavily_search import TavilySearchAPIWrapper

from backend.config import get_settings

logger = logging.getLogger(__name__)


class TavilySearchError(RuntimeError):
    """Raised when Tavily is unavailable or returns an unusable response."""


def _get_api_key() -> str:
    """Read a Tavily key from the process environment or application settings."""
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        try:
            key = get_settings().tavily_api_key.strip()
        except Exception:  # noqa: BLE001 - Settings also validates unrelated API keys.
            key = ""

    if not key or key.lower() in {
        "your_tavily_api_key",
        "your_tavily_api_key_here",
    }:
        raise TavilySearchError(
            "TAVILY_API_KEY is missing. Add a valid Tavily key to the environment or .env."
        )
    return key


def _extract_results(response: Any) -> list[dict[str, Any]]:
    """Normalize Tavily's result payload to title, URL, and content records."""
    if isinstance(response, str):
        if "no search results found" in response.lower():
            return []
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise TavilySearchError("Tavily returned an unreadable response.") from exc

    if isinstance(response, dict):
        if response.get("error"):
            raise TavilySearchError("Tavily reported a search error.")
        results = response.get("results", [])
    elif isinstance(response, list):
        results = response
    else:
        raise TavilySearchError("Tavily returned an unsupported response format.")

    if not isinstance(results, list):
        raise TavilySearchError("Tavily returned an invalid results collection.")

    normalized: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        record = {
            key: item[key]
            for key in ("title", "url", "content", "score")
            if item.get(key) is not None
        }
        if "content" not in record and item.get("raw_content"):
            record["content"] = item["raw_content"]
        if record:
            normalized.append(record)
    return normalized


def search_tavily(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the web through LangChain's Tavily integration.

    Args:
        query: A specific web-search query.
        max_results: Maximum number of Tavily results to request.

    Returns:
        Search result dictionaries containing source fields when available.
        An empty list means the search completed successfully with no results.

    Raises:
        TavilySearchError: If credentials are missing or the API fails.
        ValueError: If the query or result limit is invalid.
    """
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")
    if max_results < 1:
        raise ValueError("max_results must be at least one.")

    api_key = _get_api_key()
    try:
        wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
        tool = TavilySearch(
            api_wrapper=wrapper,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
        )
        response = tool.invoke({"query": query.strip()})
    except ToolException as exc:
        # TavilySearch raises ToolException for a normal no-results response.
        if "no search results found" in str(exc).lower():
            return []
        logger.warning("Tavily search failed with %s", type(exc).__name__)
        raise TavilySearchError("Tavily search request failed.") from exc
    except Exception as exc:  # noqa: BLE001 - external client errors vary by version.
        logger.warning("Tavily search failed with %s", type(exc).__name__)
        raise TavilySearchError("Tavily search request failed.") from exc

    return _extract_results(response)
