"""Offline tests for the LangGraph shared travel state contract."""

from backend.graph.state import (
    TravelState,
    create_initial_travel_state,
    merge_dicts,
    merge_errors,
    model_to_state_update,
)
from backend.models.schemas import Attraction, TravelRequest


def test_initial_travel_state_creation() -> None:
    state: TravelState = create_initial_travel_state()

    assert state == {"revision_count": 0, "agent_status": {}, "errors": []}


def test_state_initialization_from_travel_request() -> None:
    request = TravelRequest(
        origin="Bengaluru", destination="Kochi", budget=20000, preferences=["food"]
    )
    state = create_initial_travel_state(request, user_request="Plan a Kochi trip")

    assert state["origin"] == "Bengaluru"
    assert state["destination"] == "Kochi"
    assert state["currency"] == "INR"
    assert state["user_request"] == "Plan a Kochi trip"
    assert state["revision_count"] == 0


def test_state_supports_partial_updates_and_preserves_independent_outputs() -> None:
    state: TravelState = create_initial_travel_state(
        TravelRequest(destination="Kochi")
    )
    state.update({"attractions": [Attraction(name="Marine Drive")]})
    state.update({"travel_tips": ["Check local opening hours."]})

    assert state["attractions"][0].name == "Marine Drive"
    assert state["travel_tips"] == ["Check local opening hours."]


def test_pydantic_model_serializes_for_state_updates() -> None:
    update = model_to_state_update(Attraction(name="Marine Drive", estimated_cost=0))

    assert update == {"name": "Marine Drive", "estimated_cost": 0.0}
    assert isinstance(update, dict)


def test_dictionary_merge_reducer_is_non_mutating() -> None:
    current = {"research": "complete"}
    update = {"budget": "running", "research": "revised"}

    merged = merge_dicts(current, update)

    assert merged == {"research": "revised", "budget": "running"}
    assert current == {"research": "complete"}
    assert update == {"budget": "running", "research": "revised"}


def test_error_reducer_preserves_independent_agent_errors() -> None:
    current = ["research failed"]
    update = ["budget unavailable"]

    assert merge_errors(current, update) == ["research failed", "budget unavailable"]
    assert current == ["research failed"]
