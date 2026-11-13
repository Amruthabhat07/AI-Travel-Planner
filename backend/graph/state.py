"""Shared, partial state contract for future travel-planning agents.

Each specialist writes to its own state keys. Only independently accumulated
fields use reducers, allowing LangGraph to combine parallel node updates
without replacing unrelated outputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel

from backend.models.schemas import (
    Attraction,
    BudgetBreakdown,
    BudgetPriceInput,
    DestinationResearch,
    FinalTravelPlan,
    ItineraryDay,
    TravelRequest,
    TravelInformation,
    ValidationResult,
)


def merge_dicts(
    current: Mapping[str, str] | None, update: Mapping[str, str] | None
) -> dict[str, str]:
    """Return a new dictionary containing both updates; newer duplicate keys win.

    A fresh copy is returned so a reducer invocation never mutates a state
    dictionary owned by LangGraph or another parallel node.
    """
    merged = dict(current or {})
    merged.update(update or {})
    return merged


def merge_errors(current: list[str] | None, update: list[str] | None) -> list[str]:
    """Concatenate errors reported by independent nodes without mutating inputs."""
    return [*(current or []), *(update or [])]


class TravelState(TypedDict, total=False):
    """LangGraph-compatible shared state; total=False supports partial updates."""

    # User requirements
    user_request: str | None
    origin: str | None
    destination: str | None
    start_date: date | None
    end_date: date | None
    duration: int | None
    budget: float | None
    currency: str
    travelers: int
    preferences: list[str]
    travel_style: str | None
    additional_requirements: str | None

    # Destination research outputs
    destination_research: DestinationResearch | None
    attractions: list[Attraction]
    recommended_activities: list[str]
    destination_information: dict[str, Any] | None

    # Itinerary outputs
    itinerary: list[ItineraryDay]
    daily_schedule: dict[str, list[str]] | None
    transportation_suggestions: list[str]

    # Budget outputs
    budget_breakdown: BudgetBreakdown | None
    budget_inputs: list[BudgetPriceInput] | list[dict[str, Any]]
    emergency_buffer_percent: float | None
    estimated_total_cost: float | None
    remaining_budget: float | None
    budget_status: str | None

    # Travel assistant outputs
    weather_information: str | None
    accommodation_suggestions: list[str]
    travel_tips: list[str]
    safety_information: list[str]
    travel_information: TravelInformation | None

    # Validation outputs
    validation_result: ValidationResult | None
    validation_issues: list[str]
    improvement_suggestions: list[str]
    revision_count: int

    # Final response and operational status
    final_plan: FinalTravelPlan | None
    agent_status: Annotated[dict[str, str], merge_dicts]
    errors: Annotated[list[str], merge_errors]


def create_initial_travel_state(
    request: TravelRequest | None = None, *, user_request: str | None = None
) -> TravelState:
    """Build an initial partial state from an optional validated request.

    Agent-owned output keys are omitted until a future node produces them.
    ``user_request`` is optional because the structured request schema does not
    retain the exact natural-language wording that produced it.
    """
    state: TravelState = {
        "revision_count": 0,
        "agent_status": {},
        "errors": [],
    }
    if request is not None:
        state.update(request.model_dump(mode="python", exclude_none=True))
    if user_request is not None:
        state["user_request"] = user_request
    return state


def model_to_state_update(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model into a state-ready, recursively plain mapping."""
    return model.model_dump(mode="python", exclude_none=True)
