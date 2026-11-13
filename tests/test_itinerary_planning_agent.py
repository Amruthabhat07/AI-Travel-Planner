"""Offline tests for the itinerary planning state-update agent."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.agents import itinerary_planning_agent as itinerary_module
from backend.agents.itinerary_planning_agent import itinerary_planning_agent
from backend.models.schemas import Attraction, DestinationResearch, TravelRequest
from backend.tools.route_optimizer import assign_attractions_to_days


@pytest.fixture
def planning_state() -> dict[str, object]:
    request = TravelRequest(
        origin="Tokyo",
        destination="Kyoto",
        start_date=date(2027, 4, 1),
        end_date=date(2027, 4, 3),
        duration=3,
        budget=50000,
        currency="JPY",
        travelers=2,
        preferences=["history", "food"],
        travel_style="moderate",
        additional_requirements="Include breaks.",
    )
    research = DestinationResearch(
        destination="Kyoto",
        overview="A historic city.",
        attractions=[
            Attraction(
                name="Temple A",
                description="A historic temple.",
                latitude=35.011,
                longitude=135.768,
            ),
            Attraction(
                name="Temple B",
                description="Another nearby temple.",
                latitude=35.012,
                longitude=135.769,
            ),
            Attraction(name="Food Market", description="Local food market."),
        ],
        recommended_activities=["Try local cuisine"],
        research_sources=["https://example.test/kyoto"],
    )
    return {
        **request.model_dump(mode="python", exclude_none=True),
        "user_request": "Plan a history and food trip to Kyoto.",
        "destination_research": research.model_dump(mode="python"),
        "agent_status": {"destination_research": "completed"},
        "errors": [],
    }


def _response_days() -> list[dict[str, object]]:
    return [
        {
            "day_number": index,
            "date": date(2027, 4, index).isoformat(),
            "title": f"Kyoto day {index}",
            "activities": [
                f"Morning: Visit planned sights; Afternoon: Explore; Evening: Rest or dine; day {index}."
            ],
            "travel_notes": ["Allow time for breaks."],
        }
        for index in range(1, 4)
    ]


def _mock_llm(monkeypatch: pytest.MonkeyPatch, content: object) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=content)
    monkeypatch.setattr(itinerary_module, "get_llm", lambda: llm)
    return llm


def test_successful_itinerary_generation(planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _mock_llm(monkeypatch, json.dumps({"itinerary": _response_days()}))

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert len(update["itinerary"]) == 3
    assert update["agent_status"]["itinerary_planning"] == "completed_limited_geographic_optimization"
    assert "errors" not in update
    llm.invoke.assert_called_once()


def test_extracts_travel_requirements_and_destination_research(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _mock_llm(monkeypatch, json.dumps({"itinerary": _response_days()}))

    itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    prompt = llm.invoke.call_args.args[0]
    assert '"budget": 50000.0' in prompt
    assert '"travelers": 2' in prompt
    assert '"preferences"' in prompt and "history" in prompt and "food" in prompt
    assert "Include breaks." in prompt
    assert "Temple A" in prompt and "Food Market" in prompt
    assert "https://example.test/kyoto" in prompt


def test_correct_number_of_days_and_dates(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, json.dumps({"itinerary": _response_days()}))

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert [day["day_number"] for day in update["itinerary"]] == [1, 2, 3]
    assert [day["date"] for day in update["itinerary"]] == [
        date(2027, 4, index) for index in range(1, 4)
    ]


def test_gemini_list_content_blocks_are_parsed(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    content = [
        {"type": "thinking", "thinking": "hidden"},
        {"type": "text", "text": "```json\n" + json.dumps({"itinerary": _response_days()}) + "\n```"},
    ]
    _mock_llm(monkeypatch, content)

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert len(update["itinerary"]) == 3


def test_missing_destination_research_is_reported(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    planning_state.pop("destination_research")
    llm = MagicMock()
    monkeypatch.setattr(itinerary_module, "get_llm", lambda: llm)

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert update["agent_status"]["itinerary_planning"] == "failed"
    assert "research is missing" in update["errors"][0]
    assert "itinerary" not in update
    llm.invoke.assert_not_called()


def test_empty_research_attractions_are_rejected(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    planning_state["destination_research"] = DestinationResearch(
        destination="Kyoto"
    ).model_dump()
    llm = MagicMock()
    monkeypatch.setattr(itinerary_module, "get_llm", lambda: llm)

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert update["agent_status"]["itinerary_planning"] == "failed"
    assert "no attractions" in update["errors"][0]
    llm.invoke.assert_not_called()


def test_gemini_api_failure_returns_failure_status(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("network unavailable")
    monkeypatch.setattr(itinerary_module, "get_llm", lambda: llm)

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert update["agent_status"]["itinerary_planning"] == "failed"
    assert "Gemini itinerary generation failed" in update["errors"][0]
    assert "itinerary" not in update


def test_invalid_gemini_json_is_rejected(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, "{broken")

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert update["agent_status"]["itinerary_planning"] == "failed"
    assert "malformed itinerary JSON" in update["errors"][0]


def test_pydantic_validation_failure_is_reported(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid_days = _response_days()
    invalid_days[0]["title"] = ""
    _mock_llm(monkeypatch, json.dumps({"itinerary": invalid_days}))

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert update["agent_status"]["itinerary_planning"] == "failed"
    assert "schema validation" in update["errors"][0]


def test_incomplete_day_coverage_is_rejected(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, json.dumps({"itinerary": _response_days()[:2]}))

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert update["agent_status"]["itinerary_planning"] == "failed"
    assert "coverage is incomplete" in update["errors"][0]


def test_date_duration_inconsistency_is_rejected(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    planning_state["duration"] = 4
    llm = MagicMock()
    monkeypatch.setattr(itinerary_module, "get_llm", lambda: llm)

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert update["agent_status"]["itinerary_planning"] == "failed"
    assert "does not match" in update["errors"][0]
    llm.invoke.assert_not_called()


def test_correct_use_of_geographic_clustering_output(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _mock_llm(monkeypatch, json.dumps({"itinerary": _response_days()}))
    actual_plan = assign_attractions_to_days(
        DestinationResearch.model_validate(planning_state["destination_research"]).attractions,
        3,
    )
    cluster_names = {
        item.name
        for cluster in actual_plan.clustering.clusters
        for item in cluster
    }
    assert cluster_names == {"Temple A", "Temple B"}

    monkeypatch.setattr(
        itinerary_module,
        "assign_attractions_to_days",
        MagicMock(return_value=actual_plan),
    )
    itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    prompt = llm.invoke.call_args.args[0]
    assert "Temple A" in prompt and "Temple B" in prompt
    assert '"geographic_optimization_status": "optimized"' in prompt
    assert "straight-line" in prompt


def test_route_order_and_metadata_are_state_serialized(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, json.dumps({"itinerary": _response_days()}))

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert isinstance(update["itinerary"], list)
    assert all(isinstance(day, dict) for day in update["itinerary"])
    optimized_day = next(
        day for day in update["itinerary"] if day["geographic_optimization_status"] == "optimized"
    )
    assert set(optimized_day["attraction_order"]) == {"Temple A", "Temple B"}
    assert optimized_day["estimated_straight_line_distance_km"] >= 0
    assert all("road distance" not in note.lower() for day in update["itinerary"] for note in day["geographic_notes"])


def test_agent_status_distinguishes_no_geographic_data(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    research = DestinationResearch(
        destination="Kyoto",
        attractions=[Attraction(name="No coordinates")],
    )
    planning_state["destination_research"] = research.model_dump()
    _mock_llm(monkeypatch, json.dumps({"itinerary": _response_days()}))

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]

    assert update["agent_status"]["itinerary_planning"] == "completed_limited_geographic_optimization"
    assert all(
        day["geographic_optimization_status"] == "unavailable"
        for day in update["itinerary"]
        if day["attraction_order"]
    )


def test_unrelated_state_fields_are_preserved(
    planning_state: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_llm(monkeypatch, json.dumps({"itinerary": _response_days()}))
    planning_state["budget_status"] = "pending"
    planning_state["travel_tips"] = ["Bring a light jacket."]
    before = dict(planning_state)

    update = itinerary_planning_agent(planning_state)  # type: ignore[arg-type]
    resulting_state = {**planning_state, **update}

    assert resulting_state["budget_status"] == before["budget_status"]
    assert resulting_state["travel_tips"] == before["travel_tips"]
    assert set(update).issubset({"itinerary", "agent_status", "errors"})
