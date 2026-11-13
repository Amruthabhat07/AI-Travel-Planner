"""Itinerary planning node combining Gemini scheduling and deterministic routing."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError

from backend.graph.state import TravelState, model_to_state_update
from backend.llm import get_llm
from backend.models.schemas import DestinationResearch, ItineraryDay, TravelRequest
from backend.tools.route_optimizer import (
    DailyAttractionPlan,
    assign_attractions_to_days,
    optimize_route,
)

logger = logging.getLogger(__name__)
_AGENT_KEY = "itinerary_planning"


def _status_update(status: str, error: str | None = None) -> dict[str, Any]:
    """Build a reducer-compatible partial status/error update."""
    update: dict[str, Any] = {"agent_status": {_AGENT_KEY: status}}
    if error:
        update["errors"] = [error]
    return update


def _date_value(value: Any) -> date | None:
    """Parse optional date-like state values without accepting invalid dates."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid travel date: {value!r}.") from exc
    raise ValueError("Travel dates must be date objects or ISO date strings.")


def _request_from_state(state: Mapping[str, Any]) -> TravelRequest:
    """Validate state requirements and infer only duration/date arithmetic."""
    start = _date_value(state.get("start_date"))
    end = _date_value(state.get("end_date"))
    duration = state.get("duration")
    if duration is not None:
        if isinstance(duration, bool):
            raise ValueError("Trip duration must be a positive whole number of days.")
        if isinstance(duration, float) and not duration.is_integer():
            raise ValueError("Trip duration must be a positive whole number of days.")
        try:
            duration = int(duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("Trip duration must be a positive whole number of days.") from exc
        if duration < 1:
            raise ValueError("Trip duration must be a positive whole number of days.")

    if start is not None and end is not None:
        if end < start:
            raise ValueError("End date must not precede start date.")
        date_duration = (end - start).days + 1
        if duration is None:
            duration = date_duration
        elif duration != date_duration:
            raise ValueError("Trip duration does not match the inclusive travel dates.")
    elif duration is not None and start is not None:
        end = start + timedelta(days=duration - 1)
    elif duration is not None and end is not None:
        start = end - timedelta(days=duration - 1)

    if duration is None:
        raise ValueError("Provide a positive duration or both travel dates.")

    request_data = {
        "origin": state.get("origin"),
        "destination": state.get("destination"),
        "start_date": start,
        "end_date": end,
        "duration": duration,
        "budget": state.get("budget"),
        "currency": state.get("currency", "INR"),
        "travelers": state.get("travelers", 1),
        "preferences": state.get("preferences") or [],
        "travel_style": state.get("travel_style"),
        "additional_requirements": state.get("additional_requirements"),
    }
    try:
        return TravelRequest.model_validate(request_data)
    except ValidationError as exc:
        raise ValueError(f"Travel requirements are invalid: {exc}") from exc


def _research_from_state(state: Mapping[str, Any], destination: str) -> DestinationResearch:
    """Validate research stored by the earlier destination research node."""
    raw_research = state.get("destination_research")
    if raw_research is None:
        raise ValueError("Destination research is missing from TravelState.")
    try:
        research = DestinationResearch.model_validate(raw_research)
    except ValidationError as exc:
        raise ValueError(f"Destination research in TravelState is invalid: {exc}") from exc
    if research.destination.strip().casefold() != destination.casefold():
        raise ValueError("Destination research does not match the requested destination.")
    if not research.attractions:
        raise ValueError("Destination research contains no attractions to schedule.")
    return research


def _daily_route_data(plan: DailyAttractionPlan) -> list[dict[str, Any]]:
    """Produce each day's proposed visit order and honest straight-line metadata."""
    daily_data: list[dict[str, Any]] = []
    for day_index, candidates in enumerate(plan.days, start=1):
        located = [item for item in candidates if item.latitude is not None and item.longitude is not None]
        missing = [item for item in candidates if item.latitude is None or item.longitude is None]
        if located:
            route = optimize_route(located)
            ordered = [*route.ordered_attractions, *missing]
            status = "limited" if missing else "optimized"
            distance = route.distance_km
        else:
            ordered = list(candidates)
            status = "unavailable"
            distance = None
        daily_data.append(
            {
                "day_number": day_index,
                "attractions": [item.model_dump(mode="json", exclude_none=True) for item in ordered],
                "attraction_order": [item.name for item in ordered],
                "geographic_optimization_status": status,
                "estimated_straight_line_distance_km": distance,
                "geographic_notes": (
                    [f"Geographic ordering unavailable for: {', '.join(item.name for item in missing)}."]
                    if missing
                    else []
                ),
            }
        )
    return daily_data


def _build_prompt(
    request: TravelRequest,
    research: DestinationResearch,
    daily_routes: list[dict[str, Any]],
    user_request: str | None,
) -> str:
    """Ask Gemini for day schedules using deterministic geographic guidance."""
    schema = ItineraryDay.model_json_schema()
    context = request.model_dump(mode="json", exclude_none=True)
    guidance = {
        "user_request": user_request,
        "travel_requirements": context,
        "destination_research": research.model_dump(mode="json", exclude_none=True),
        "daily_geographic_candidates": daily_routes,
    }
    return f"""You are an itinerary planner. Return only JSON with this shape:
{{"itinerary": [<one object per travel day>]}}
Each item must validate against this JSON Schema:
{json.dumps(schema, ensure_ascii=False)}

Plan exactly {request.duration} consecutive travel days for {request.destination}.
Use the supplied research and traveler preferences. Include every day's
attraction_order in activities in the stated visiting order, unless a clear
scheduling reason requires an adjustment; when adjusting, explain the reason in
travel_notes and avoid unnecessary backtracking. The supplied route sequence is
based on deterministic Haversine straight-line estimates, not road routes or
travel times. Do not invent opening hours, ticket prices, transit durations,
road distances, or coordinates. Leave estimated_daily_cost null unless a cost
is supported by the supplied research. Build a realistic morning, afternoon,
and evening schedule with suitable meal/rest breaks; do not assume every day
needs evening activities. Include all day numbers exactly once. Use the specified
calendar date for each day, or null when dates were not supplied or derivable.
For attractions without coordinates, include them as candidates but do not claim
they were geographically optimized. Use available preferences, group size,
budget, travel style, origin, and additional requirements. Do not invent a hotel
or a start point if none is provided.

Planning data:
{json.dumps(guidance, ensure_ascii=False, default=str)}

Return valid JSON only, without markdown fences or commentary."""


def _extract_text(content: Any) -> str:
    """Extract textual content from both string and structured-block responses."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False, default=str)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in {"thinking", "executable_code", "code_execution_result"}:
                    continue
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _parse_itinerary_response(response: Any) -> list[dict[str, Any]]:
    """Read an itinerary list from raw JSON, mappings, or Gemini response blocks."""
    if isinstance(response, dict):
        payload: Any = response
    else:
        content = _extract_text(getattr(response, "content", response)).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
        if not content:
            raise ValueError("Gemini returned empty itinerary content.")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned malformed itinerary JSON.") from exc

    if isinstance(payload, list):
        days = payload
    elif isinstance(payload, dict):
        days = payload.get("itinerary")
    else:
        days = None
    if not isinstance(days, list) or not all(isinstance(item, dict) for item in days):
        raise ValueError("Gemini itinerary output must contain a list of day objects.")
    return days


def _validate_days(
    raw_days: list[dict[str, Any]], request: TravelRequest, route_data: list[dict[str, Any]]
) -> list[ItineraryDay]:
    """Validate exact day coverage/dates and attach route metadata computed locally."""
    duration = request.duration
    if duration is None:
        raise ValueError("Trip duration is unavailable.")
    by_number: dict[int, dict[str, Any]] = {}
    for raw_day in raw_days:
        try:
            day_number = int(raw_day.get("day_number"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Every itinerary day must have an integer day_number.") from exc
        if day_number in by_number:
            raise ValueError(f"Gemini returned duplicate day number {day_number}.")
        by_number[day_number] = raw_day

    expected_numbers = set(range(1, duration + 1))
    if set(by_number) != expected_numbers:
        missing = sorted(expected_numbers - set(by_number))
        unexpected = sorted(set(by_number) - expected_numbers)
        raise ValueError(
            f"Itinerary day coverage is incomplete (missing={missing}, unexpected={unexpected})."
        )

    validated: list[ItineraryDay] = []
    for day_number in range(1, duration + 1):
        raw_day = dict(by_number[day_number])
        expected_date = (
            request.start_date + timedelta(days=day_number - 1)
            if request.start_date is not None
            else None
        )
        supplied_date = _date_value(raw_day.get("date"))
        if supplied_date != expected_date:
            raise ValueError(
                f"Itinerary day {day_number} date must be {expected_date.isoformat() if expected_date else 'null'}."
            )

        route = route_data[day_number - 1]
        # Route claims are derived from coordinates here, not from Gemini output.
        raw_day.update(
            {
                "attraction_order": route["attraction_order"],
                "geographic_optimization_status": route["geographic_optimization_status"],
                "estimated_straight_line_distance_km": route[
                    "estimated_straight_line_distance_km"
                ],
                "geographic_notes": route["geographic_notes"],
            }
        )
        try:
            validated.append(ItineraryDay.model_validate(raw_day))
        except ValidationError as exc:
            raise ValueError(f"Itinerary day {day_number} failed schema validation: {exc}") from exc
    return validated


def itinerary_planning_agent(state: TravelState) -> dict[str, Any]:
    """Generate and validate a day-by-day itinerary as a partial state update.

    Nearby researched attractions are grouped with DBSCAN and routed with
    nearest-neighbor plus 2-opt based on straight-line Haversine distance. Gemini
    produces the actual schedule while those computed routes act as planning
    guidance. The output only writes ``itinerary`` and this agent's status/errors.
    """
    try:
        request = _request_from_state(state)
        research = _research_from_state(state, request.destination)
        daily_plan = assign_attractions_to_days(research.attractions, request.duration or 0)
        route_data = _daily_route_data(daily_plan)
    except (ValueError, ValidationError, TypeError) as exc:
        return _status_update("failed", str(exc))

    prompt = _build_prompt(request, research, route_data, state.get("user_request"))
    try:
        response = get_llm().invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - provider exceptions vary by release.
        logger.exception("Gemini itinerary generation failed")
        return _status_update(
            "failed", f"Gemini itinerary generation failed ({type(exc).__name__})."
        )

    try:
        raw_days = _parse_itinerary_response(response)
        itinerary = _validate_days(raw_days, request, route_data)
    except (ValidationError, ValueError, TypeError) as exc:
        logger.warning("Gemini itinerary output was invalid: %s", exc)
        return _status_update("failed", f"Itinerary output is invalid: {exc}")

    located_count = len(daily_plan.clustering.noise) + sum(
        len(cluster) for cluster in daily_plan.clustering.clusters
    )
    unlocated_count = len(daily_plan.clustering.without_coordinates)
    if located_count == 0:
        status = "completed_limited_geographic_optimization"
    elif unlocated_count:
        status = "completed_limited_geographic_optimization"
    else:
        status = "completed_with_geographic_optimization"

    return {
        "itinerary": [model_to_state_update(day) for day in itinerary],
        "agent_status": {_AGENT_KEY: status},
    }
