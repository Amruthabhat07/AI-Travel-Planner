"""Offline tests for deterministic validation rules and the state-update agent."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from backend.agents.budget_planning_agent import calculate_budget
from backend.agents.validation_agent import validation_agent, validate_travel_state
from backend.models.schemas import (
    AdviceItem,
    DestinationResearch,
    ItineraryDay,
    PackingRecommendations,
    TravelInformation,
    TravelRequest,
    ValidationResult,
    TransportationInformation,
    WeatherReport,
)


def _price(category: str, name: str, amount: str, unit: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "category": category,
        "name": name,
        "unit_price": amount,
        "currency": "INR",
        "pricing_unit": unit,
        "price_type": "user_provided",
        "verification_status": "user_supplied",
        "source_name": "Test user input",
        **kwargs,
    }


def _valid_budget() -> dict[str, Any]:
    state = {
        "destination": "Pune",
        "start_date": date(2027, 2, 1),
        "end_date": date(2027, 2, 2),
        "duration": 2,
        "travelers": 2,
        "budget": 5000,
        "currency": "INR",
        "emergency_buffer_percent": 10,
        "budget_inputs": [
            _price("accommodation", "Room", "100", "per_room_per_night", rooms=1, nights=1),
            _price("food", "Meals", "20", "per_person_per_day"),
            _price("local_transportation", "Transit", "10", "per_day"),
            _price("intercity_transportation", "Train", "50", "per_person_ticket"),
            _price("activity", "Museum tickets", "5", "per_person_ticket", visits=1),
            _price("other_expenses", "Insurance", "10", "flat"),
        ],
    }
    return calculate_budget(state).model_dump(mode="python", exclude_none=False)


def _valid_research() -> dict[str, Any]:
    return DestinationResearch(
        destination="Pune",
        overview="A researched destination overview.",
        best_time_to_visit="As indicated in research.",
        attractions=[
            {"name": "City Museum", "description": "A researched museum.", "latitude": 18.5204, "longitude": 73.8567},
            {"name": "Garden", "description": "A researched garden."},
        ],
        recommended_activities=["Visit the museum"],
        local_culture="Research-backed cultural notes.",
        research_sources=["https://example.test/pune"],
    ).model_dump(mode="python", exclude_none=False)


def _valid_itinerary() -> list[dict[str, Any]]:
    return [
        ItineraryDay(
            day_number=1,
            date=date(2027, 2, 1),
            title="Museum day",
            activities=["Visit City Museum"],
            attraction_order=["City Museum"],
            estimated_straight_line_distance_km=2.3,
        ).model_dump(mode="python", exclude_none=False),
        ItineraryDay(
            day_number=2,
            date=date(2027, 2, 2),
            title="Garden day",
            activities=["Visit Garden"],
            attraction_order=["Garden"],
        ).model_dump(mode="python", exclude_none=False),
    ]


def _valid_travel_information() -> dict[str, Any]:
    source = {
        "title": "Open-Meteo forecast",
        "url": "https://open-meteo.com/en/docs",
        "provider": "Open-Meteo",
        "retrieved_at": "2026-11-13T10:00:00+00:00",
        "information_type": "forecast",
    }
    general = AdviceItem(text="Confirm current details before travel.", basis="general_guidance")
    return TravelInformation(
        weather_summary="Forecast available for the trip dates.",
        temperature="15–25 °C",
        weather=WeatherReport(
            status="forecast",
            location="Pune",
            retrieved_at="2026-11-13T10:00:00+00:00",
            daily_forecasts=[
                {"date": "2027-02-01", "condition": "Clear sky", "temperature_min_c": 15, "temperature_max_c": 25, "precipitation_mm": 0, "source": source},
                {"date": "2027-02-02", "condition": "Partly cloudy", "temperature_min_c": 16, "temperature_max_c": 24, "precipitation_mm": 0, "source": source},
            ],
        ),
        transportation=TransportationInformation(
            status="general_guidance",
            local_options=[general],
            intercity_options=[general],
            logistics_tips=[general],
        ),
        packing=PackingRecommendations(
            clothing=[general], footwear=[general], personal_essentials=[general],
            travel_documents=[general], electronics_accessories=[general],
            activity_specific=[general], forecast_based=False,
        ),
        practical_tips=[general],
        sources=[source],
    ).model_dump(mode="python", exclude_none=False)


def _valid_state(**overrides: Any) -> dict[str, Any]:
    request = TravelRequest(
        origin="Mumbai",
        destination="Pune",
        start_date=date(2027, 2, 1),
        end_date=date(2027, 2, 2),
        duration=2,
        budget=5000,
        currency="INR",
        travelers=2,
        preferences=["history"],
    )
    state: dict[str, Any] = {
        **request.model_dump(mode="python", exclude_none=True),
        "destination_research": _valid_research(),
        "itinerary": _valid_itinerary(),
        "budget_breakdown": _valid_budget(),
        "travel_information": _valid_travel_information(),
        "agent_status": {
            "destination_research": "completed",
            "itinerary_planning": "completed_with_geographic_optimization",
            "budget_planning": "completed",
            "travel_assistant": "completed",
        },
        "errors": [],
    }
    state.update(overrides)
    return state


def _find(result: Any, code: str) -> list[Any]:
    return [issue for issue in result.issues if issue.code == code]


def test_fully_valid_travel_state_passes_all_major_categories() -> None:
    result = validate_travel_state(_valid_state())  # type: ignore[arg-type]

    assert result.status == "PASSED"
    assert result.is_valid is True
    assert result.error_count == result.warning_count == 0
    assert result.checks_performed == result.checks_passed
    assert result.budget_within_limit is True
    assert result.itinerary_feasible is True
    assert {issue.category for issue in result.issues} == {
        "travel_request", "destination_research", "itinerary", "budget", "travel_information"
    }


def test_valid_travel_request_is_recorded_as_pass() -> None:
    result = validate_travel_state(_valid_state())  # type: ignore[arg-type]
    assert _find(result, "request_schema_valid")[0].severity == "PASS"


def test_missing_destination_is_a_request_error() -> None:
    state = _valid_state(destination="")
    result = validate_travel_state(state)  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert _find(result, "request_invalid")


def test_invalid_traveler_count_is_a_request_error() -> None:
    result = validate_travel_state(_valid_state(travelers=0))  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert _find(result, "request_invalid")


def test_invalid_dates_are_reported() -> None:
    result = validate_travel_state(_valid_state(start_date="not-a-date"))  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert _find(result, "request_invalid")


def test_inconsistent_duration_and_dates_are_reported() -> None:
    result = validate_travel_state(_valid_state(duration=3))  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert _find(result, "request_invalid")


def test_valid_destination_research_passes() -> None:
    result = validate_travel_state(_valid_state())  # type: ignore[arg-type]
    assert _find(result, "research_schema_valid")[0].severity == "PASS"
    assert _find(result, "research_destination_matches")[0].severity == "PASS"


def test_missing_research_fields_are_warnings_or_errors_by_field() -> None:
    result = validate_travel_state(
        _valid_state(destination_research={"destination": "Pune", "attractions": []})
    )  # type: ignore[arg-type]
    assert _find(result, "research_overview_missing")[0].severity == "WARNING"
    assert _find(result, "attractions_missing")[0].severity == "ERROR"


def test_invalid_attraction_coordinates_fail_schema_validation() -> None:
    research = _valid_research()
    research["attractions"][0]["latitude"] = 91
    result = validate_travel_state(_valid_state(destination_research=research))  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert _find(result, "research_invalid")


def test_attraction_cost_without_price_source_warns() -> None:
    research = _valid_research()
    research["attractions"][0]["estimated_cost"] = 5
    result = validate_travel_state(_valid_state(destination_research=research))  # type: ignore[arg-type]
    assert _find(result, "attraction_cost_source_missing")[0].severity == "WARNING"


def test_duplicate_attractions_are_errors() -> None:
    research = _valid_research()
    research["attractions"].append({"name": " city museum ", "description": "Duplicate"})
    result = validate_travel_state(_valid_state(destination_research=research))  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert _find(result, "duplicate_attraction")


def test_valid_itinerary_passes_schema_and_dates() -> None:
    result = validate_travel_state(_valid_state())  # type: ignore[arg-type]
    assert _find(result, "itinerary_schema_valid")
    assert _find(result, "itinerary_duration_matches")
    assert not _find(result, "itinerary_date_mismatch")


def test_incorrect_itinerary_duration_is_error() -> None:
    result = validate_travel_state(_valid_state(itinerary=_valid_itinerary()[:1]))  # type: ignore[arg-type]
    assert _find(result, "itinerary_duration_mismatch")


def test_duplicate_itinerary_day_numbers_are_errors() -> None:
    itinerary = _valid_itinerary()
    itinerary[1]["day_number"] = 1
    result = validate_travel_state(_valid_state(itinerary=itinerary))  # type: ignore[arg-type]
    assert _find(result, "itinerary_day_number_duplicate")


def test_duplicate_itinerary_dates_are_errors() -> None:
    itinerary = _valid_itinerary()
    itinerary[1]["date"] = date(2027, 2, 1)
    result = validate_travel_state(_valid_state(itinerary=itinerary))  # type: ignore[arg-type]
    assert _find(result, "itinerary_date_duplicate")


def test_itinerary_date_mismatch_is_error() -> None:
    itinerary = _valid_itinerary()
    itinerary[0]["date"] = date(2027, 2, 2)
    result = validate_travel_state(_valid_state(itinerary=itinerary))  # type: ignore[arg-type]
    assert _find(result, "itinerary_date_mismatch")


def test_invalid_attraction_reference_is_error() -> None:
    itinerary = _valid_itinerary()
    itinerary[0]["attraction_order"] = ["Unknown Place"]
    result = validate_travel_state(_valid_state(itinerary=itinerary))  # type: ignore[arg-type]
    assert _find(result, "attraction_reference_invalid")


def test_not_visiting_every_researched_attraction_is_not_an_error() -> None:
    itinerary = _valid_itinerary()
    itinerary[1]["attraction_order"] = []
    result = validate_travel_state(_valid_state(itinerary=itinerary))  # type: ignore[arg-type]
    assert not _find(result, "attraction_reference_invalid")
    assert _find(result, "attraction_order_missing")[0].severity == "WARNING"


def test_valid_budget_is_independently_recalculated() -> None:
    result = validate_travel_state(_valid_state())  # type: ignore[arg-type]
    assert not [issue for issue in result.issues if issue.category == "budget" and issue.severity == "ERROR"]
    assert result.budget_within_limit is True


def test_incorrect_budget_subtotal_is_error() -> None:
    budget = _valid_budget()
    budget["known_subtotal"] = Decimal("1.00")
    result = validate_travel_state(_valid_state(budget_breakdown=budget))  # type: ignore[arg-type]
    assert _find(result, "budget_subtotal_mismatch")


def test_incorrect_emergency_buffer_is_error() -> None:
    budget = _valid_budget()
    budget["emergency_buffer"] = Decimal("1.00")
    result = validate_travel_state(_valid_state(budget_breakdown=budget))  # type: ignore[arg-type]
    assert _find(result, "emergency_buffer_mismatch")


def test_incorrect_final_total_is_error() -> None:
    budget = _valid_budget()
    budget["total_estimated_cost"] += Decimal("1.00")
    result = validate_travel_state(_valid_state(budget_breakdown=budget))  # type: ignore[arg-type]
    assert _find(result, "budget_total_mismatch")


def test_incorrect_cost_per_person_is_error() -> None:
    budget = _valid_budget()
    budget["estimated_cost_per_person"] += Decimal("1.00")
    result = validate_travel_state(_valid_state(budget_breakdown=budget))  # type: ignore[arg-type]
    assert _find(result, "cost_per_person_mismatch")


def test_incorrect_remaining_budget_is_error() -> None:
    budget = _valid_budget()
    budget["remaining_budget"] += Decimal("1.00")
    result = validate_travel_state(_valid_state(budget_breakdown=budget))  # type: ignore[arg-type]
    assert _find(result, "remaining_budget_mismatch")


def test_mixed_currency_line_items_are_detected() -> None:
    budget = _valid_budget()
    budget["line_items"][0]["currency"] = "USD"
    result = validate_travel_state(_valid_state(budget_breakdown=budget))  # type: ignore[arg-type]
    assert _find(result, "budget_mixed_currencies")


def test_negative_budget_line_item_is_invalid() -> None:
    budget = _valid_budget()
    budget["line_items"][0]["unit_price"] = Decimal("-1")
    result = validate_travel_state(_valid_state(budget_breakdown=budget))  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert _find(result, "budget_invalid")


def test_partial_budget_produces_warning_not_false_success() -> None:
    partial_state = {
        **_valid_state(),
        "budget_inputs": [_price("food", "Meals", "20", "per_person_per_day")],
    }
    partial_state["budget_breakdown"] = calculate_budget(partial_state).model_dump(mode="python", exclude_none=False)
    result = validate_travel_state(partial_state)  # type: ignore[arg-type]
    assert _find(result, "budget_estimate_partial")
    assert result.budget_within_limit is None


def test_unverified_price_included_in_total_is_error() -> None:
    budget = _valid_budget()
    item = budget["line_items"][0]
    item["verification_status"] = "unverified"
    result = validate_travel_state(_valid_state(budget_breakdown=budget))  # type: ignore[arg-type]
    assert _find(result, "unverified_line_included")
    assert _find(result, "budget_subtotal_mismatch")


def test_excluded_unverified_price_is_not_recalculated_into_totals() -> None:
    state = {
        "destination": "Pune",
        "start_date": date(2027, 2, 1),
        "end_date": date(2027, 2, 2),
        "duration": 2,
        "travelers": 2,
        "budget": 5000,
        "currency": "INR",
        "budget_inputs": [
            _price("food", "Unverified meals", "20", "per_person_per_day", price_type="published", verification_status="unverified", source_url="https://example.test/price")
        ],
        "emergency_buffer_percent": 0,
    }
    state["budget_breakdown"] = calculate_budget(state).model_dump(mode="python", exclude_none=False)
    state["agent_status"] = {"budget_planning": "completed_partial"}

    result = validate_travel_state(state)  # type: ignore[arg-type]

    assert not _find(result, "unverified_line_included")
    assert _find(result, "budget_estimate_partial")
    assert result.budget_within_limit is None


def test_missing_budget_breakdown_is_warning_and_partial() -> None:
    state = _valid_state()
    state.pop("budget_breakdown")
    state["agent_status"].pop("budget_planning")
    result = validate_travel_state(state)  # type: ignore[arg-type]
    assert result.status == "PARTIAL"
    assert _find(result, "budget_missing")[0].severity == "WARNING"


def test_missing_weather_forecast_is_warning() -> None:
    info = _valid_travel_information()
    info["weather"] = None
    result = validate_travel_state(_valid_state(travel_information=info))  # type: ignore[arg-type]
    assert _find(result, "weather_missing")[0].severity == "WARNING"
    assert result.status == "PASSED_WITH_WARNINGS"


def test_missing_travel_information_is_warning_when_assistant_did_not_succeed() -> None:
    state = _valid_state()
    state.pop("travel_information")
    state["agent_status"].pop("travel_assistant")
    result = validate_travel_state(state)  # type: ignore[arg-type]
    assert _find(result, "travel_information_missing")[0].severity == "WARNING"
    assert result.status == "PARTIAL"


def test_invalid_travel_information_structure_is_error() -> None:
    info = _valid_travel_information()
    info["transportation"] = {"status": "invalid"}
    result = validate_travel_state(_valid_state(travel_information=info))  # type: ignore[arg-type]
    assert _find(result, "travel_information_invalid")


def test_forecast_dates_outside_trip_are_errors() -> None:
    info = _valid_travel_information()
    info["weather"]["daily_forecasts"][0]["date"] = date(2027, 2, 3)
    result = validate_travel_state(_valid_state(travel_information=info))  # type: ignore[arg-type]
    assert _find(result, "weather_date_outside_trip")


def test_forecast_based_packing_without_weather_is_error() -> None:
    info = _valid_travel_information()
    info["weather"] = None
    info["packing"]["forecast_based"] = True
    info["packing"]["weather_specific"] = [
        {"text": "Rain jacket", "basis": "forecast", "sources": []}
    ]
    result = validate_travel_state(_valid_state(travel_information=info))  # type: ignore[arg-type]
    assert _find(result, "packing_forecast_without_weather")


def test_invalid_travel_source_url_is_error() -> None:
    info = _valid_travel_information()
    info["sources"][0]["url"] = "not-a-url"
    result = validate_travel_state(_valid_state(travel_information=info))  # type: ignore[arg-type]
    assert _find(result, "travel_source_url_invalid")


def test_exact_transport_values_are_flagged_as_not_independently_verified() -> None:
    info = _valid_travel_information()
    info["transportation"]["local_options"] = [
        {
            "text": "The bus costs ₹50 and leaves at 08:30.",
            "basis": "verified_source",
            "sources": [{
                "title": "Transit guide",
                "url": "https://example.test/transit",
                "provider": "Publisher",
                "information_type": "web_research",
            }],
        }
    ]

    result = validate_travel_state(_valid_state(travel_information=info))  # type: ignore[arg-type]

    assert _find(result, "exact_transport_claim_not_independently_verifiable")


def test_warning_only_result_has_warning_status_and_counts() -> None:
    info = _valid_travel_information()
    info["weather"] = None
    result = validate_travel_state(_valid_state(travel_information=info))  # type: ignore[arg-type]
    assert result.status == "PASSED_WITH_WARNINGS"
    assert result.is_valid is True
    assert result.warning_count > 0
    assert result.error_count == 0
    assert result.checks_performed == len(result.issues)


def test_error_result_is_not_passed_and_counts_match() -> None:
    result = validate_travel_state(_valid_state(destination=""))  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert result.is_valid is False
    assert result.error_count == sum(issue.severity == "ERROR" for issue in result.issues)


def test_missing_required_agent_outputs_yield_partial_status() -> None:
    result = validate_travel_state({"destination": "Pune"})  # type: ignore[arg-type]
    assert result.status == "PARTIAL"
    assert result.error_count == 0
    assert result.warning_count > 0


def test_validation_agent_writes_only_validation_and_status() -> None:
    state = _valid_state()
    before = dict(state)
    update = validation_agent(state)  # type: ignore[arg-type]

    assert set(update) == {"validation_result", "agent_status"}
    assert update["validation_result"]["status"] == "PASSED"
    assert update["agent_status"]["validation_agent"] == "completed"
    assert state == before


def test_validation_preserves_all_existing_agent_outputs() -> None:
    state = _valid_state()
    before = {key: state[key] for key in ("destination_research", "itinerary", "budget_breakdown", "travel_information")}

    update = validation_agent(state)  # type: ignore[arg-type]

    assert {key: state[key] for key in before} == before
    assert not set(update).intersection(before)


def test_agent_handles_unexpected_validation_failure_without_mutating_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _valid_state()

    def fail_validation(current_state: dict[str, Any]) -> Any:
        assert current_state["destination"] == "Pune"
        raise RuntimeError("unexpected")

    monkeypatch.setattr("backend.agents.validation_agent.validate_travel_state", fail_validation)

    update = validation_agent(state)  # type: ignore[arg-type]

    assert update["agent_status"]["validation_agent"] == "failed"
    assert "unexpected" in update["errors"][0]
    assert "validation_result" not in update
    assert "budget_breakdown" in state


def test_validation_agent_has_no_llm_import_or_calls() -> None:
    from backend.agents import validation_agent as module

    assert not hasattr(module, "get_llm")
    assert validate_travel_state(_valid_state()).status == "PASSED"  # type: ignore[arg-type]


def test_zero_budget_is_reported_by_canonical_travel_request_schema() -> None:
    result = validate_travel_state(_valid_state(budget=0))  # type: ignore[arg-type]
    assert result.status == "FAILED"
    assert _find(result, "request_invalid")


def test_legacy_validation_result_string_issues_are_upgraded() -> None:
    legacy = ValidationResult(is_valid=True, issues=["Optional field was missing"])

    assert legacy.status == "PASSED_WITH_WARNINGS"
    assert legacy.warning_count == 1
    assert legacy.checks_performed == 1
    assert legacy.issues[0].message == "Optional field was missing"


def test_legacy_invalid_validation_result_gets_error_status() -> None:
    legacy = ValidationResult(is_valid=False)

    assert legacy.status == "FAILED"
    assert legacy.error_count == 1
    assert legacy.issues[0].severity == "ERROR"
