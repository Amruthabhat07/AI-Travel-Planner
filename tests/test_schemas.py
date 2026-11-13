"""Offline tests for the reusable travel request and output schemas."""

from datetime import date

import pytest
from pydantic import ValidationError

from backend.models.schemas import (
    Attraction,
    BudgetBreakdown,
    DestinationResearch,
    FinalTravelPlan,
    ItineraryDay,
    TravelInformation,
    TravelRequest,
)


def test_valid_travel_request_creation() -> None:
    request = TravelRequest(
        origin="Bengaluru",
        destination="Kochi",
        start_date="2026-12-01",
        end_date="2026-12-03",
        duration=3,
        budget=25000,
        preferences=["food", "history"],
    )

    assert request.destination == "Kochi"
    assert request.currency == "INR"
    assert request.start_date == date(2026, 12, 1)


def test_missing_destination_validation() -> None:
    with pytest.raises(ValidationError):
        TravelRequest.model_validate({})


def test_invalid_budget_validation() -> None:
    with pytest.raises(ValidationError):
        TravelRequest(destination="Kochi", budget=0)


def test_invalid_traveler_count() -> None:
    with pytest.raises(ValidationError):
        TravelRequest(destination="Kochi", travelers=0)


def test_invalid_travel_dates() -> None:
    with pytest.raises(ValidationError):
        TravelRequest(
            destination="Kochi", start_date="2026-12-05", end_date="2026-12-01"
        )


def test_valid_attraction_creation() -> None:
    attraction = Attraction(
        name="Marine Drive",
        description="Waterfront promenade.",
        location="Kochi",
        category="promenade",
    )

    assert attraction.estimated_cost is None
    assert attraction.name == "Marine Drive"


def test_attraction_optional_coordinates_are_range_checked_and_paired() -> None:
    attraction = Attraction(name="Temple", latitude=35.0, longitude=135.0)
    assert attraction.latitude == 35.0
    assert attraction.longitude == 135.0

    with pytest.raises(ValidationError):
        Attraction(name="Invalid latitude", latitude=91, longitude=0)
    with pytest.raises(ValidationError):
        Attraction(name="Unpaired coordinates", latitude=35)


def test_valid_itinerary_day_creation() -> None:
    day = ItineraryDay(
        day_number=1,
        date="2026-12-01",
        title="Arrival day",
        activities=["Check in"],
    )

    assert day.date == date(2026, 12, 1)
    assert day.estimated_daily_cost is None
    assert day.geographic_optimization_status is None


def test_valid_budget_breakdown_creation() -> None:
    budget = BudgetBreakdown(
        accommodation_cost=100,
        food_cost=50,
        transportation_cost=25,
        activity_cost=20,
        miscellaneous_cost=5,
        total_estimated_cost=200,
        remaining_budget=50,
    )

    assert budget.currency == "INR"
    assert budget.total_estimated_cost == 200


def test_valid_final_travel_plan_creation() -> None:
    plan = FinalTravelPlan(
        destination="Kochi",
        trip_summary="A short trip.",
        itinerary=[ItineraryDay(day_number=1, title="Explore", activities=["Walk"])],
        destination_research=DestinationResearch(destination="Kochi"),
        budget_breakdown=BudgetBreakdown(total_estimated_cost=100),
        travel_information=TravelInformation(weather_summary="Not yet researched"),
        important_tips=["Confirm opening hours."],
        total_estimated_cost=100,
    )

    assert plan.destination == "Kochi"
    assert isinstance(plan.itinerary[0], ItineraryDay)


def test_invalid_data_rejection() -> None:
    with pytest.raises(ValidationError):
        ItineraryDay(day_number=0, title="Invalid day")

    with pytest.raises(ValidationError):
        BudgetBreakdown(
            accommodation_cost=100,
            food_cost=20,
            transportation_cost=10,
            activity_cost=5,
            miscellaneous_cost=5,
            total_estimated_cost=200,
        )


def test_date_duration_must_be_consistent() -> None:
    with pytest.raises(ValidationError, match="Duration must match"):
        TravelRequest(
            destination="Kochi",
            start_date="2026-12-01",
            end_date="2026-12-03",
            duration=2,
        )
