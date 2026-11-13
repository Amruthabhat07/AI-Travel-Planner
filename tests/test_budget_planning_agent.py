"""Offline tests for deterministic, Decimal-based budget calculations."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from backend.agents.budget_planning_agent import budget_planning_agent, calculate_budget
from backend.models.schemas import BudgetPriceInput


def _price(
    category: str,
    name: str,
    amount: str | int | float,
    unit: str,
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "category": category,
        "name": name,
        "unit_price": amount,
        "currency": "INR",
        "pricing_unit": unit,
        "price_type": "user_provided",
        "verification_status": "user_supplied",
        "source_name": "User supplied",
        **kwargs,
    }


def _base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "destination": "Pune",
        "start_date": date(2027, 2, 1),
        "end_date": date(2027, 2, 3),
        "duration": 3,
        "travelers": 2,
        "budget": 1000,
        "currency": "INR",
        "budget_inputs": [
            _price("accommodation", "Room", "100", "per_room_per_night", rooms=2, nights=3),
            _price("food", "Meals", "10", "per_person_per_day"),
            _price("local_transportation", "Local pass", "15", "per_day"),
            _price("intercity_transportation", "Train tickets", "50", "per_person_ticket"),
            _price("activity", "Museum tickets", "5", "per_person_ticket", visits=2),
            _price("other_expenses", "Insurance", "30", "flat"),
        ],
        "emergency_buffer_percent": 10,
    }
    state.update(overrides)
    return state


def test_correct_accommodation_calculation() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.accommodation_cost == Decimal("600.00")
    item = next(item for item in breakdown.line_items if item.category == "accommodation")
    assert item.quantity == Decimal("6")


def test_correct_food_calculation() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.food_cost == Decimal("60.00")


def test_correct_local_transportation_calculation() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.local_transportation_cost == Decimal("45.00")


def test_local_transport_trip_unit_uses_explicit_trip_count() -> None:
    state = _base_state(
        budget_inputs=[_price("local_transportation", "Taxi", "12.50", "per_trip", trips=4)]
    )
    breakdown = calculate_budget(state)  # type: ignore[arg-type]
    assert breakdown.local_transportation_cost == Decimal("50.00")


def test_correct_intercity_transportation_calculation() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.intercity_transportation_cost == Decimal("100.00")


def test_intercity_vehicle_price_uses_explicit_vehicle_count() -> None:
    state = _base_state(
        budget_inputs=[
            _price("intercity_transportation", "Van", "500", "per_vehicle", vehicles=2)
        ]
    )
    breakdown = calculate_budget(state)  # type: ignore[arg-type]
    assert breakdown.intercity_transportation_cost == Decimal("1000.00")


def test_correct_activity_ticket_calculation() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.activity_cost == Decimal("20.00")


def test_adult_and_child_ticket_prices_use_separate_counts() -> None:
    adult = _price(
        "activity", "Adult tickets", "10", "per_person_ticket", applicable_travelers=2, visits=1
    )
    child = _price(
        "activity", "Child tickets", "5", "per_person_ticket", applicable_travelers=1, visits=1
    )
    breakdown = calculate_budget(_base_state(budget_inputs=[adult, child]))  # type: ignore[arg-type]
    assert breakdown.activity_cost == Decimal("25.00")


def test_correct_other_expenses_calculation() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.miscellaneous_cost == Decimal("30.00")


def test_known_subtotal_is_sum_of_expense_categories() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.known_subtotal == Decimal("855.00")


def test_emergency_buffer_and_final_total() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.emergency_buffer_percent == Decimal("10")
    assert breakdown.emergency_buffer == Decimal("85.50")
    assert breakdown.total_estimated_cost == Decimal("940.50")


def test_per_person_cost_remaining_and_utilization() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.estimated_cost_per_person == Decimal("470.25")
    assert breakdown.remaining_budget == Decimal("59.50")
    assert breakdown.budget_utilization_percent == Decimal("94.05")


def test_complete_budget_within_user_limit() -> None:
    breakdown = calculate_budget(_base_state())  # type: ignore[arg-type]
    assert breakdown.estimate_status == "complete"
    assert breakdown.budget_comparison_status == "within_limit"


def test_budget_exceeding_limit_is_reported() -> None:
    breakdown = calculate_budget(_base_state(budget=500))  # type: ignore[arg-type]
    assert breakdown.budget_comparison_status == "over_budget"
    assert breakdown.remaining_budget == Decimal("-440.50")


def test_missing_categories_are_unknown_not_zero() -> None:
    breakdown = calculate_budget(_base_state(budget_inputs=[]))  # type: ignore[arg-type]
    assert breakdown.estimate_status == "partial"
    assert breakdown.total_estimated_cost is None
    assert breakdown.known_subtotal == Decimal("0.00")
    assert breakdown.accommodation_cost is None
    assert set(breakdown.missing_categories) == {
        "accommodation", "food", "local_transportation", "intercity_transportation", "activity", "other_expenses"
    }


def test_partial_estimate_comparison_is_provisional() -> None:
    inputs = [_price("food", "Meals", "10", "per_person_per_day")]
    breakdown = calculate_budget(_base_state(budget_inputs=inputs))  # type: ignore[arg-type]
    assert breakdown.food_cost == Decimal("60.00")
    assert breakdown.estimate_status == "partial"
    assert breakdown.budget_comparison_status == "provisional"
    assert breakdown.remaining_budget == Decimal("934.00")


def test_different_currencies_are_rejected_without_conversion() -> None:
    state = _base_state(
        budget_inputs=[_price("food", "Meals", "10", "per_person_per_day", currency="USD")]
    )
    with pytest.raises(ValueError, match="Currency mismatch"):
        calculate_budget(state)  # type: ignore[arg-type]


def test_negative_price_is_rejected_by_schema_and_agent() -> None:
    with pytest.raises(ValidationError):
        BudgetPriceInput.model_validate(_price("food", "Meals", "-1", "per_person_per_day"))
    bad = _price("food", "Meals", "-1", "per_person_per_day")
    update = budget_planning_agent(_base_state(budget_inputs=[bad]))  # type: ignore[arg-type]
    assert update["agent_status"]["budget_planning"] == "failed"
    assert "Budget price input is invalid" in update["errors"][0]


def test_invalid_or_missing_quantities_are_rejected() -> None:
    with pytest.raises(ValueError, match="explicit rooms and nights"):
        calculate_budget(
            _base_state(budget_inputs=[_price("accommodation", "Room", 100, "per_room_per_night", nights=2)])
        )  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BudgetPriceInput.model_validate(
            _price("local_transportation", "Taxi", 10, "per_trip", trips=-1)
        )
    with pytest.raises(ValueError, match="explicit trip count"):
        calculate_budget(
            _base_state(budget_inputs=[_price("local_transportation", "Taxi", 10, "per_trip")])
        )  # type: ignore[arg-type]


def test_missing_travelers_is_an_agent_failure() -> None:
    state = _base_state()
    state.pop("travelers")
    update = budget_planning_agent(state)  # type: ignore[arg-type]
    assert update["agent_status"]["budget_planning"] == "failed"
    assert "number of travelers is required" in update["errors"][0]


def test_zero_travelers_is_rejected() -> None:
    update = budget_planning_agent(_base_state(travelers=0))  # type: ignore[arg-type]
    assert update["agent_status"]["budget_planning"] == "failed"
    assert "Travel requirements are invalid" in update["errors"][0]


def test_zero_budget_fails_safely_without_division() -> None:
    update = budget_planning_agent(_base_state(budget=0))  # type: ignore[arg-type]
    assert update["agent_status"]["budget_planning"] == "failed"
    assert "Travel requirements are invalid" in update["errors"][0]


def test_decimal_precision_and_currency_rounding() -> None:
    inputs = [
        _price("food", "Meal A", "0.10", "per_person_per_day"),
        _price("other_expenses", "Small item", "0.105", "flat"),
    ]
    breakdown = calculate_budget(_base_state(budget_inputs=inputs, emergency_buffer_percent="0"))  # type: ignore[arg-type]
    assert breakdown.food_cost == Decimal("0.60")
    assert breakdown.miscellaneous_cost == Decimal("0.11")
    assert breakdown.known_subtotal == Decimal("0.71")
    assert breakdown.total_estimated_cost == Decimal("0.71")


def test_user_provided_source_metadata_is_preserved() -> None:
    breakdown = calculate_budget(_base_state(budget_inputs=[_price("food", "Meals", 10, "per_person_per_day")]))  # type: ignore[arg-type]
    line = breakdown.line_items[0]
    assert line.price_type == "user_provided"
    assert line.source_name == "User supplied"
    assert line.verification_status == "user_supplied"


def test_verified_published_source_metadata_is_preserved() -> None:
    published = _price(
        "food", "Restaurant menu", 20, "per_person_per_day",
        price_type="published", verification_status="verified",
        source_name="Restaurant menu", source_url="https://example.test/menu",
    )
    breakdown = calculate_budget(_base_state(budget_inputs=[published]))  # type: ignore[arg-type]
    line = breakdown.line_items[0]
    assert line.price_type == "published"
    assert line.source_url == "https://example.test/menu"
    assert line.verification_status == "verified"


def test_unverified_published_price_is_retained_but_not_calculated() -> None:
    unverified = _price(
        "food", "Unverified menu", 20, "per_person_per_day",
        price_type="published", verification_status="unverified",
        source_name="Restaurant menu", source_url="https://example.test/menu",
    )
    breakdown = calculate_budget(_base_state(budget_inputs=[unverified]))  # type: ignore[arg-type]
    line = breakdown.line_items[0]
    assert line.included_in_total is False
    assert line.calculated_amount is None
    assert line.source_url == "https://example.test/menu"
    assert "not verified" in line.exclusion_reason
    assert "food" in breakdown.missing_categories
    assert breakdown.total_estimated_cost is None


def test_sourced_destination_research_ticket_price_is_used() -> None:
    state = _base_state(budget_inputs=[])
    state["destination_research"] = {
        "destination": "Pune",
        "attractions": [
            {
                "name": "Museum",
                "estimated_cost": 25,
                "estimated_cost_currency": "INR",
                "estimated_cost_pricing_unit": "per_person_ticket",
                "estimated_cost_source_name": "Official museum tariff",
                "estimated_cost_source_url": "https://example.test/tariffs",
                "estimated_cost_price_type": "published",
                "estimated_cost_verification_status": "verified",
            }
        ],
    }

    breakdown = calculate_budget(state)  # type: ignore[arg-type]

    research_ticket = next(item for item in breakdown.line_items if item.name == "Attraction ticket: Museum")
    assert research_ticket.calculated_amount == Decimal("50.00")
    assert research_ticket.source_url == "https://example.test/tariffs"


def test_unsourced_destination_research_cost_is_not_treated_as_reliable() -> None:
    state = _base_state(budget_inputs=[])
    state["destination_research"] = {
        "destination": "Pune",
        "attractions": [{"name": "Museum", "estimated_cost": 25}],
    }

    breakdown = calculate_budget(state)  # type: ignore[arg-type]

    assert "activity" in breakdown.missing_categories
    assert breakdown.line_items == []


def test_approved_estimate_is_visible_in_estimate_status() -> None:
    estimate = _price(
        "food", "Manually approved meals estimate", 20, "per_person_per_day",
        price_type="estimate", verification_status="approved",
    )
    breakdown = calculate_budget(_base_state(budget_inputs=[estimate]))  # type: ignore[arg-type]
    assert breakdown.estimate_status == "partial_with_assumptions"
    assert breakdown.line_items[0].price_type == "estimate"


def test_unapproved_estimate_is_excluded_and_kept_for_review() -> None:
    estimate = _price(
        "food", "Unapproved meal estimate", 20, "per_person_per_day",
        price_type="estimate", verification_status="manual_review",
    )
    breakdown = calculate_budget(_base_state(budget_inputs=[estimate]))  # type: ignore[arg-type]
    assert breakdown.line_items[0].included_in_total is False
    assert "not been manually approved" in breakdown.line_items[0].exclusion_reason
    assert "food" in breakdown.missing_categories


def test_local_and_intercity_transport_are_not_double_counted() -> None:
    inputs = [
        _price("local_transportation", "City pass", 10, "per_day"),
        _price("intercity_transportation", "Rail ticket", 20, "per_person_ticket"),
    ]
    breakdown = calculate_budget(_base_state(budget_inputs=inputs))  # type: ignore[arg-type]
    assert breakdown.local_transportation_cost == Decimal("30.00")
    assert breakdown.intercity_transportation_cost == Decimal("40.00")
    assert breakdown.transportation_cost == Decimal("70.00")
    assert breakdown.known_subtotal == Decimal("70.00")


def test_duplicate_budget_lines_are_rejected_to_prevent_double_counting() -> None:
    duplicate_inputs = [
        _price("food", "Meals", 10, "per_person_per_day"),
        _price("food", "Meals", 10, "per_person_per_day"),
    ]
    with pytest.raises(ValueError, match="Duplicate budget input"):
        calculate_budget(_base_state(budget_inputs=duplicate_inputs))  # type: ignore[arg-type]


def test_agent_updates_budget_fields_and_preserves_every_other_field() -> None:
    state = _base_state()
    state.update({"destination_research": {"destination": "Pune"}, "itinerary": ["unchanged"], "travel_tips": ["tip"]})
    update = budget_planning_agent(state)  # type: ignore[arg-type]

    assert update["agent_status"]["budget_planning"] == "completed"
    assert update["budget_breakdown"]["total_estimated_cost"] == Decimal("940.50")
    assert set(update) == {"budget_breakdown", "agent_status"}
    resulting_state = {**state, **update}
    assert resulting_state["destination_research"] == state["destination_research"]
    assert resulting_state["itinerary"] == state["itinerary"]
    assert resulting_state["travel_tips"] == state["travel_tips"]


def test_agent_failure_does_not_return_partial_or_fake_budget() -> None:
    state = _base_state()
    state["currency"] = "USD"
    update = budget_planning_agent(state)  # type: ignore[arg-type]
    assert update["agent_status"]["budget_planning"] == "failed"
    assert "errors" in update
    assert "budget_breakdown" not in update


def test_invalid_buffer_percentage_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        calculate_budget(_base_state(emergency_buffer_percent="101"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        calculate_budget(_base_state(emergency_buffer_percent="NaN"))  # type: ignore[arg-type]


def test_zero_buffer_is_explicit_and_does_not_hide_a_default() -> None:
    breakdown = calculate_budget(_base_state(emergency_buffer_percent=0))  # type: ignore[arg-type]
    assert breakdown.emergency_buffer_percent == Decimal("0")
    assert breakdown.emergency_buffer == Decimal("0.00")
