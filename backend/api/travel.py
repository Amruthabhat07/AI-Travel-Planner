"""REST endpoints that expose the existing travel-planning workflow."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import Field, ValidationError

from backend.graph.workflow import invoke_travel_workflow
from backend.models.schemas import (
    BudgetBreakdown, BudgetPriceInput, DestinationResearch, FinalTravelPlan,
    ItineraryDay, TravelInformation, TravelRequest, ValidationResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/travel", tags=["Travel planning"])


class TravelPlanRequest(TravelRequest):
    """Existing validated travel requirements plus supported workflow options."""

    user_request: str | None = Field(default=None, description="Optional original free-text request.")
    budget_inputs: list[BudgetPriceInput] | None = Field(default=None, description="Explicit source-aware budget prices.")
    emergency_buffer_percent: float | None = Field(default=None, ge=0, le=100)


class TravelPlanResponse(FinalTravelPlan):
    """Final plan plus specialist outputs useful to API clients."""

    request: TravelRequest
    destination_research: DestinationResearch | None = None
    itinerary: list[ItineraryDay] = Field(default_factory=list)
    budget_breakdown: BudgetBreakdown | None = None
    travel_information: TravelInformation | None = None
    validation_result: ValidationResult | None = None
    agent_status: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


def _as_model(value: Any, model_type: type[Any]) -> Any | None:
    return None if value is None else model_type.model_validate(value)


def _response_from_state(state: dict[str, Any], request: TravelRequest) -> TravelPlanResponse:
    """Validate state-owned response data before FastAPI serializes it."""
    try:
        final_plan = _as_model(state.get("final_plan"), FinalTravelPlan)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Workflow returned an invalid final plan: %s", type(exc).__name__)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Travel planning returned an invalid response.") from exc
    if final_plan is None:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Travel planning completed without a final response.")
    try:
        data = final_plan.model_dump(mode="python")
        data.update({
            "request": request,
            "destination_research": _as_model(state.get("destination_research"), DestinationResearch),
            "itinerary": [ItineraryDay.model_validate(item) for item in state.get("itinerary", [])],
            "budget_breakdown": _as_model(state.get("budget_breakdown"), BudgetBreakdown),
            "travel_information": _as_model(state.get("travel_information"), TravelInformation),
            "validation_result": _as_model(state.get("validation_result"), ValidationResult),
            "agent_status": dict(state.get("agent_status") or {}),
            "errors": [str(error) for error in state.get("errors", [])],
        })
        return TravelPlanResponse.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - malformed workflow output is an upstream failure.
        logger.warning("Workflow returned an invalid response shape: %s", type(exc).__name__)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Travel planning returned an invalid response.") from exc


@router.post("/plan", response_model=TravelPlanResponse)
def generate_travel_plan(payload: TravelPlanRequest) -> TravelPlanResponse:
    """Validate a request, invoke LangGraph, and serialize its final plan."""
    request = TravelRequest.model_validate(payload.model_dump(exclude={"user_request", "budget_inputs", "emergency_buffer_percent"}))
    try:
        state = invoke_travel_workflow(request, user_request=payload.user_request, budget_inputs=payload.budget_inputs, emergency_buffer_percent=payload.emergency_buffer_percent)
    except (ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never expose workflow tracebacks through HTTP.
        logger.exception("Travel workflow execution failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Travel planning could not be completed.") from exc
    if not isinstance(state, dict):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Travel planning returned an invalid response.")
    return _response_from_state(state, request)


@router.get("/status")
def workflow_status() -> dict[str, str | bool]:
    """Return static service availability without compiling or invoking the graph."""
    return {"status": "available", "workflow_available": True, "service": "ai-travel-planner"}
