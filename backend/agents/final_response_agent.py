"""Python-only final formatting for the travel-planning shared state."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.graph.state import TravelState, model_to_state_update
from backend.models.schemas import BudgetBreakdown, DestinationResearch, FinalTravelPlan, ItineraryDay, TravelInformation, TravelRequest, ValidationResult

_AGENT_KEY = "final_response_generator"


def _plain(value: Any) -> Any:
    return value.model_dump(mode="python", exclude_none=False) if isinstance(value, BaseModel) else value


def _parse(value: Any, cls: type[BaseModel]) -> Any | None:
    if value is None:
        return None
    try:
        return cls.model_validate(_plain(value))
    except ValidationError:
        return None


def _clean(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        value = _clean(value)
        if value and value not in result:
            result.append(value)
    return result


def _money(value: Any, currency: str) -> str | None:
    """Show an upstream value exactly; never calculate, round, or convert it."""
    return None if value is None else f"{currency} {value}"


def _section(lines: list[str], heading: str) -> None:
    lines.extend(("", f"## {heading}", ""))


def _advice(items: Iterable[Any]) -> list[str]:
    return [item.text for item in items if _clean(getattr(item, "text", None))]


def _sources(research: DestinationResearch | None, info: TravelInformation | None) -> list[str]:
    urls: list[str] = []
    if research:
        urls.extend(research.research_sources)
        urls.extend(a.estimated_cost_source_url for a in research.attractions if a.estimated_cost_source_url)
    if info:
        urls.extend(s.url for s in info.sources if s.url)
        if info.weather:
            urls.extend(day.source.url for day in info.weather.daily_forecasts if day.source.url)
        collections: list[Iterable[Any]] = [info.practical_tips]
        if info.transportation:
            collections += [info.transportation.local_options, info.transportation.intercity_options, info.transportation.logistics_tips]
        if info.packing:
            collections += [info.packing.clothing, info.packing.footwear, info.packing.personal_essentials, info.packing.travel_documents, info.packing.electronics_accessories, info.packing.weather_specific, info.packing.activity_specific]
        for collection in collections:
            for item in collection:
                urls.extend(s.url for s in item.sources if s.url)
    return _unique(urls)


def generate_final_response(state: TravelState) -> FinalTravelPlan:
    """Build a useful partial response without inventing absent agent output."""
    research = _parse(state.get("destination_research"), DestinationResearch)
    budget = _parse(state.get("budget_breakdown"), BudgetBreakdown)
    info = _parse(state.get("travel_information"), TravelInformation)
    validation = _parse(state.get("validation_result"), ValidationResult)
    request_fields = ("origin", "destination", "start_date", "end_date", "duration", "budget", "currency", "travelers", "preferences", "travel_style", "additional_requirements")
    request = _parse({key: state[key] for key in request_fields if key in state}, TravelRequest)
    raw_days = state.get("itinerary")
    try:
        itinerary = [ItineraryDay.model_validate(_plain(day)) for day in raw_days] if isinstance(raw_days, list) else None
    except ValidationError:
        itinerary = None

    destination = request.destination if request else None
    currency = request.currency if request else (budget.currency if budget else "INR")
    missing: list[str] = []
    for value, message in ((research, "Destination research is unavailable."), (itinerary, "Day-wise itinerary is unavailable."), (budget, "Budget breakdown could not be generated."), (info, "Weather and travel information is unavailable."), (validation, "Validation report is unavailable.")):
        if value is None:
            missing.append(message)
    if budget and budget.estimate_status != "complete":
        categories = ", ".join(item.replace("_", " ") for item in budget.missing_categories)
        missing.append("Budget estimate is incomplete" + (f"; missing categories: {categories}." if categories else "."))

    notes = list(missing)
    if research:
        notes.extend(f"Entry fee for {a.name} is unverified and is not presented as a confirmed price." for a in research.attractions if a.estimated_cost is not None and a.estimated_cost_verification_status not in {"verified", "approved"})
    if validation:
        notes.extend(issue.message for issue in validation.issues if issue.severity in {"WARNING", "ERROR"})
    notes.extend(error for error in state.get("errors", []) if _clean(error))
    notes = _unique(notes)

    lines = ["# Travel Plan"]
    _section(lines, "Trip Overview")
    lines.append(f"- **Destination:** {destination}" if destination else "- Destination was not available in the validated request.")
    for label, value in (("Travellers", request.travelers if request else None), ("Start date", request.start_date if request else None), ("End date", request.end_date if request else None), ("Total trip duration", request.duration if request else None)):
        if value is not None:
            lines.append(f"- **{label}:** {value}{' days' if label == 'Total trip duration' else ''}")
    if request and request.budget is not None:
        lines.append(f"- **Travel budget:** {_money(request.budget, currency)}")

    _section(lines, "Destination Overview")
    if research:
        for label, value in (("Description", research.overview), ("Best time to visit", research.best_time_to_visit), ("Local culture, food, and practical information", research.local_culture)):
            if value:
                lines.append(f"- **{label}:** {value}")
        if research.recommended_activities:
            lines.append("- **Recommended activities:** " + "; ".join(research.recommended_activities))
    else:
        lines.append("Destination research is unavailable.")

    _section(lines, "Recommended Attractions")
    if research and research.attractions:
        for attraction in research.attractions:
            lines.append(f"### {attraction.name}")
            for label, value in (("Description", attraction.description), ("Location", attraction.location), ("Estimated visit duration", attraction.recommended_duration)):
                if value:
                    lines.append(f"- **{label}:** {value}")
            if attraction.estimated_cost is not None and attraction.estimated_cost_verification_status in {"verified", "approved"}:
                lines.append(f"- **Entry fee (verified):** {_money(attraction.estimated_cost, attraction.estimated_cost_currency or currency)}")
            elif attraction.estimated_cost is not None:
                lines.append("- **Entry fee:** Available in research but unverified; confirm before booking.")
            if attraction.estimated_cost_source_url:
                lines.append(f"- **Source:** {attraction.estimated_cost_source_url}")
    else:
        lines.append("No researched attractions are available.")

    _section(lines, "Day-wise Itinerary")
    if itinerary:
        for day in sorted(itinerary, key=lambda item: (item.day_number, item.date or date.max)):
            lines.append(f"### Day {day.day_number}" + (f" — {day.date.isoformat()}" if day.date else ""))
            lines.append(f"- **Plan:** {day.title}")
            if day.activities:
                slots: dict[str, list[str]] = {"Morning": [], "Afternoon": [], "Evening": []}
                unslotted: list[str] = []
                for activity in day.activities:
                    prefix, separator, detail = activity.partition(":")
                    label = prefix.strip().title()
                    if separator and label in slots and detail.strip():
                        slots[label].append(detail.strip())
                    else:
                        unslotted.append(activity)
                for label in ("Morning", "Afternoon", "Evening"):
                    if slots[label]:
                        lines.append(f"- **{label}:** " + "; ".join(slots[label]))
                if unslotted:
                    lines.append("- **Activities:** " + "; ".join(unslotted))
            if day.attraction_order:
                lines.append("- **Attractions:** " + " → ".join(day.attraction_order))
            if day.estimated_straight_line_distance_km is not None:
                lines.append(f"- **Route note:** Approx. straight-line route distance: {day.estimated_straight_line_distance_km} km.")
            lines.extend(f"- {note}" for note in [*day.travel_notes, *day.geographic_notes])
    else:
        lines.append("Day-wise itinerary is unavailable; no substitute activities were created.")

    _section(lines, "Budget Breakdown")
    if budget:
        for label, value in (("Accommodation", budget.accommodation_cost), ("Food", budget.food_cost), ("Local transportation", budget.local_transportation_cost), ("Intercity transportation", budget.intercity_transportation_cost), ("Activities and tickets", budget.activity_cost), ("Other expenses", budget.miscellaneous_cost), ("Emergency buffer", budget.emergency_buffer), ("Known subtotal", budget.known_subtotal), ("Final estimated total", budget.total_estimated_cost), ("Per-person cost", budget.estimated_cost_per_person), ("Remaining budget", budget.remaining_budget)):
            if (rendered := _money(value, budget.currency)) is not None:
                lines.append(f"- **{label}:** {rendered}")
        if budget.estimate_status != "complete":
            categories = ", ".join(item.replace("_", " ") for item in budget.missing_categories) or "unspecified categories"
            lines.append(f"- **Incomplete estimate:** This is a {budget.estimate_status.replace('_', ' ')} budget; missing: {categories}.")
    else:
        lines.append("Budget breakdown could not be generated.")

    _section(lines, "Weather and Travel Information")
    if info:
        if info.weather_summary:
            lines.append(f"- **Weather:** {info.weather_summary}")
        if info.weather:
            lines.append(f"- **{'Actual forecast' if info.weather.daily_forecasts else 'Forecast availability'}:** {info.weather.status.replace('_', ' ')}.")
            for forecast in info.weather.daily_forecasts:
                values = [forecast.condition, f"{forecast.temperature_min_c}–{forecast.temperature_max_c} °C" if forecast.temperature_min_c is not None and forecast.temperature_max_c is not None else None]
                lines.append(f"  - {forecast.date}: " + "; ".join(value for value in values if value))
            if info.weather.unavailable_reason:
                lines.append(f"- **Forecast limitation:** {info.weather.unavailable_reason}")
        transport = info.transport_options + (_advice(info.transportation.local_options + info.transportation.intercity_options + info.transportation.logistics_tips) if info.transportation else [])
        packing = info.packing_recommendations + (_advice(info.packing.clothing + info.packing.footwear + info.packing.personal_essentials + info.packing.travel_documents + info.packing.electronics_accessories + info.packing.weather_specific + info.packing.activity_specific) if info.packing else [])
        tips = info.safety_tips + _advice(info.practical_tips)
        if transport: lines.append("- **Transportation:** " + "; ".join(_unique(transport)))
        if packing: lines.append("- **Packing recommendations:** " + "; ".join(_unique(packing)))
        if tips: lines.append("- **Safety and practical tips:** " + "; ".join(_unique(tips)))
    else:
        lines.append("Weather and travel information is unavailable.")

    _section(lines, "Validation Report")
    if validation:
        lines.extend((f"- **Status:** {validation.status}", f"- **Total checks:** {validation.checks_performed}", f"- **Passed checks:** {validation.checks_passed}", f"- **Warnings:** {validation.warning_count}", f"- **Errors:** {validation.error_count}"))
        if validation.status in {"FAILED", "PARTIAL"}:
            lines.append("- **Attention:** This travel plan needs attention or contains incomplete information.")
        lines.extend(f"- **{issue.severity}:** {issue.message}" for issue in validation.issues if issue.severity in {"WARNING", "ERROR"})
    else:
        lines.append("Validation report is unavailable.")

    _section(lines, "Important Notes")
    lines.extend(f"- {note}" for note in notes) if notes else lines.append("No additional warnings were recorded.")
    sources = _sources(research, info)
    _section(lines, "Sources")
    lines.extend(f"- {url}" for url in sources) if sources else lines.append("No source URLs are available.")
    return FinalTravelPlan(destination=destination, trip_summary=research.overview if research and research.overview else "", itinerary=itinerary or [], destination_research=research, budget_breakdown=budget, travel_information=info, important_tips=notes, total_estimated_cost=budget.total_estimated_cost if budget else None, currency=currency, generation_status="complete" if not missing else "partial", markdown_response="\n".join(lines).strip() + "\n", missing_information=_unique(missing), validation_result=validation, source_urls=sources)


def final_response_agent(state: TravelState) -> dict[str, Any]:
    """Write only final-plan output/status, preserving every specialist output."""
    try:
        plan = generate_final_response(state)
    except Exception as exc:  # noqa: BLE001
        return {"agent_status": {_AGENT_KEY: "failed"}, "errors": [f"Final response generation failed ({type(exc).__name__}): {exc}"]}
    return {"final_plan": model_to_state_update(plan), "agent_status": {_AGENT_KEY: "completed" if plan.generation_status == "complete" else "completed_partial"}}
