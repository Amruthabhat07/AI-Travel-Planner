"""Offline tests for Tavily/Gemini destination research integration."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.agents import destination_research_agent as research_agent_module
from backend.agents.destination_research_agent import destination_research_agent
from backend.graph.state import create_initial_travel_state
from backend.models.schemas import TravelRequest
from backend.tools import search_tool
from backend.tools.search_tool import TavilySearchError


def _research_payload() -> dict[str, object]:
   return {
      "destination": "Kyoto",
      "overview": "A historic Japanese city, according to the supplied sources.",
      "best_time_to_visit": "Spring, as stated by the supplied source.",
      "attractions": [
         {
            "name": "Kiyomizu-dera",
            "description": "A historic temple.",
            "location": "Kyoto",
            "category": "temple",
            "estimated_cost": 500,
            "recommended_duration": "Two hours",
         }
      ],
      "recommended_activities": ["Explore historic neighborhoods"],
      "local_culture": "Cuisine: local dishes are described in the sources. Safety: follow local guidance.",
      "research_sources": ["https://example.test"],
   }


def _travel_state() -> dict[str, object]:
   return create_initial_travel_state(
      TravelRequest(
         origin="Tokyo",
         destination="Kyoto",
         start_date=date(2027, 4, 1),
         end_date=date(2027, 4, 3),
         duration=3,
         preferences=["food", "history"],
      ),
      user_request="Plan a food and history trip to Kyoto.",
   )


def _mock_model(monkeypatch: pytest.MonkeyPatch, content: object) -> MagicMock:
   llm = MagicMock()
   llm.invoke.return_value = SimpleNamespace(content=content)
   monkeypatch.setattr(research_agent_module, "get_llm", lambda: llm)
   return llm


def test_successful_destination_research(monkeypatch: pytest.MonkeyPatch) -> None:
   sources = [{"title": "Kyoto guide", "url": "https://example.test", "content": "Evidence."}]
   search = MagicMock(return_value=sources)
   monkeypatch.setattr(research_agent_module, "search_tavily", search)
   llm = _mock_model(monkeypatch, json.dumps(_research_payload()))

   update = destination_research_agent(_travel_state())

   assert update["destination_research"]["destination"] == "Kyoto"
   assert update["agent_status"] == {"destination_research": "completed"}
   assert "errors" not in update
   assert search.call_count == 4
   llm.invoke.assert_called_once()


def test_agent_uses_destination_and_trip_context(monkeypatch: pytest.MonkeyPatch) -> None:
   search = MagicMock(return_value=[{"title": "Source", "url": "https://example.test", "content": "Text"}])
   monkeypatch.setattr(research_agent_module, "search_tavily", search)
   llm = _mock_model(monkeypatch, json.dumps(_research_payload()))
   state = _travel_state()

   destination_research_agent(state)  # type: ignore[arg-type]

   queries = " ".join(call.args[0] for call in search.call_args_list)
   prompt = llm.invoke.call_args.args[0]
   assert "Kyoto" in queries
   assert "2027-04-01" in queries
   assert "food, history" in queries
   assert '"origin": "Tokyo"' in prompt
   assert "Plan a food and history trip to Kyoto." in prompt


def test_tavily_failure_returns_failure_status(monkeypatch: pytest.MonkeyPatch) -> None:
   monkeypatch.setattr(
      research_agent_module,
      "search_tavily",
      MagicMock(side_effect=TavilySearchError("TAVILY_API_KEY is missing.")),
   )
   llm = MagicMock()
   monkeypatch.setattr(research_agent_module, "get_llm", lambda: llm)

   update = destination_research_agent(_travel_state())  # type: ignore[arg-type]

   assert update["agent_status"]["destination_research"] == "failed"
   assert "TAVILY_API_KEY is missing" in update["errors"][0]
   assert "destination_research" not in update
   llm.invoke.assert_not_called()


def test_missing_api_key_is_reported_without_api_call(
   monkeypatch: pytest.MonkeyPatch,
) -> None:
   monkeypatch.delenv("TAVILY_API_KEY", raising=False)
   monkeypatch.setattr(search_tool, "get_settings", MagicMock(side_effect=RuntimeError))
   monkeypatch.setattr(
      search_tool,
      "TavilySearchAPIWrapper",
      MagicMock(side_effect=AssertionError("Tavily client must not be constructed")),
   )

   update = destination_research_agent(_travel_state())  # type: ignore[arg-type]

   assert update["agent_status"]["destination_research"] == "failed"
   assert "TAVILY_API_KEY is missing" in update["errors"][0]
   search_tool.TavilySearchAPIWrapper.assert_not_called()


def test_search_tool_uses_langchain_tavily_and_normalizes_sources(
   monkeypatch: pytest.MonkeyPatch,
) -> None:
   monkeypatch.setenv("TAVILY_API_KEY", "mock-tavily-key")
   wrapper = MagicMock()
   tool = MagicMock()
   tool.invoke.return_value = {
      "query": "Kyoto temples",
      "results": [
         {
            "title": "Temple guide",
            "url": "https://example.test/temples",
            "content": "Research snippet",
            "score": 0.9,
            "ignored": "not part of the normalized source shape",
         }
      ],
   }
   wrapper_factory = MagicMock(return_value=wrapper)
   tool_factory = MagicMock(return_value=tool)
   monkeypatch.setattr(search_tool, "TavilySearchAPIWrapper", wrapper_factory)
   monkeypatch.setattr(search_tool, "TavilySearch", tool_factory)

   results = search_tool.search_tavily("Kyoto temples", max_results=3)

   wrapper_factory.assert_called_once_with(tavily_api_key="mock-tavily-key")
   tool_factory.assert_called_once_with(
      api_wrapper=wrapper,
      max_results=3,
      search_depth="advanced",
      include_answer=False,
   )
   tool.invoke.assert_called_once_with({"query": "Kyoto temples"})
   assert results == [
      {
         "title": "Temple guide",
         "url": "https://example.test/temples",
         "content": "Research snippet",
         "score": 0.9,
      }
   ]


def test_search_tool_surfaces_tavily_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
   monkeypatch.setenv("TAVILY_API_KEY", "mock-tavily-key")
   tool = MagicMock()
   tool.invoke.return_value = {"error": RuntimeError("mock API failure")}
   monkeypatch.setattr(search_tool, "TavilySearchAPIWrapper", MagicMock())
   monkeypatch.setattr(search_tool, "TavilySearch", MagicMock(return_value=tool))

   with pytest.raises(TavilySearchError, match="Tavily reported a search error"):
      search_tool.search_tavily("Kyoto temples")


def test_empty_search_results_do_not_create_research(
   monkeypatch: pytest.MonkeyPatch,
) -> None:
   monkeypatch.setattr(research_agent_module, "search_tavily", MagicMock(return_value=[]))
   llm = MagicMock()
   monkeypatch.setattr(research_agent_module, "get_llm", lambda: llm)

   update = destination_research_agent(_travel_state())  # type: ignore[arg-type]

   assert update["agent_status"]["destination_research"] == "failed"
   assert "no search results" in update["errors"][0].lower()
   assert "destination_research" not in update
   llm.invoke.assert_not_called()


def test_gemini_list_content_blocks_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
   monkeypatch.setattr(
      research_agent_module,
      "search_tavily",
      MagicMock(return_value=[{"title": "Source", "url": "https://example.test", "content": "Evidence"}]),
   )
   content = [
      {"type": "thinking", "thinking": "Private reasoning omitted"},
      {"type": "text", "text": "```json\n" + json.dumps(_research_payload()) + "\n```"},
   ]
   _mock_model(monkeypatch, content)

   update = destination_research_agent(_travel_state())  # type: ignore[arg-type]

   assert update["destination_research"]["attractions"][0]["name"] == "Kiyomizu-dera"
   assert update["agent_status"]["destination_research"] == "completed"


def test_malformed_gemini_json_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
   monkeypatch.setattr(
      research_agent_module,
      "search_tavily",
      MagicMock(return_value=[{"title": "Source", "url": "https://example.test", "content": "Evidence"}]),
   )
   _mock_model(monkeypatch, "{not valid JSON")

   update = destination_research_agent(_travel_state())  # type: ignore[arg-type]

   assert update["agent_status"]["destination_research"] == "failed"
   assert "malformed JSON" in update["errors"][0]
   assert "destination_research" not in update


def test_pydantic_validation_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
   monkeypatch.setattr(
      research_agent_module,
      "search_tavily",
      MagicMock(return_value=[{"title": "Source", "url": "https://example.test", "content": "Evidence"}]),
   )
   invalid_payload = _research_payload()
   invalid_payload["attractions"] = [{"name": "Invalid", "estimated_cost": -2}]
   _mock_model(monkeypatch, json.dumps(invalid_payload))

   update = destination_research_agent(_travel_state())  # type: ignore[arg-type]

   assert update["agent_status"]["destination_research"] == "failed"
   assert "estimated_cost" in update["errors"][0]
   assert "destination_research" not in update


def test_result_is_serialized_before_state_update(monkeypatch: pytest.MonkeyPatch) -> None:
   monkeypatch.setattr(
      research_agent_module,
      "search_tavily",
      MagicMock(return_value=[{"title": "Source", "url": "https://example.test", "content": "Evidence"}]),
   )
   _mock_model(monkeypatch, json.dumps(_research_payload()))

   update = destination_research_agent(_travel_state())  # type: ignore[arg-type]

   assert isinstance(update["destination_research"], dict)
   assert isinstance(update["destination_research"]["attractions"][0], dict)
   assert update["destination_research"]["research_sources"] == ["https://example.test"]


def test_partial_update_preserves_unrelated_state_fields(
   monkeypatch: pytest.MonkeyPatch,
) -> None:
   monkeypatch.setattr(
      research_agent_module,
      "search_tavily",
      MagicMock(return_value=[{"title": "Source", "url": "https://example.test", "content": "Evidence"}]),
   )
   _mock_model(monkeypatch, json.dumps(_research_payload()))
   state = _travel_state()
   state["itinerary"] = [{"day_number": 1, "title": "Existing itinerary"}]
   state["budget_status"] = "pending"

   update = destination_research_agent(state)  # type: ignore[arg-type]
   resulting_state = {**state, **update}

   assert resulting_state["itinerary"] == state["itinerary"]
   assert resulting_state["budget_status"] == "pending"
   assert set(update).issubset({"destination_research", "agent_status", "errors"})
