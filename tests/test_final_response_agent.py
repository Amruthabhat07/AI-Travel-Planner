"""Offline tests for deterministic Phase 10 final response formatting."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from backend.agents.final_response_agent import final_response_agent, generate_final_response


def _state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "destination": "Pune", "start_date": date(2027, 2, 1), "end_date": date(2027, 2, 2),
        "duration": 2, "travelers": 2, "budget": 5000, "currency": "INR", "errors": ["Prior error is preserved."],
        "destination_research": {"destination": "Pune", "overview": "A cultural city.", "local_culture": "Try local food.", "research_sources": ["https://example.test/research"], "attractions": [{"name": "Museum", "description": "History.", "location": "Centre", "recommended_duration": "2 hours", "estimated_cost": 100, "estimated_cost_currency": "INR", "estimated_cost_verification_status": "unverified", "estimated_cost_source_url": "https://example.test/price"}]},
        "itinerary": [{"day_number": 2, "date": date(2027, 2, 2), "title": "Second", "activities": ["Evening walk"], "attraction_order": ["Museum"]}, {"day_number": 1, "date": date(2027, 2, 1), "title": "First", "activities": ["Morning museum"], "attraction_order": ["Museum"]}],
        "budget_breakdown": {"accommodation_cost": Decimal("100.00"), "food_cost": Decimal("200.00"), "activity_cost": Decimal("0.00"), "currency": "INR", "known_subtotal": Decimal("300.00"), "emergency_buffer": Decimal("0.00"), "total_estimated_cost": Decimal("300.00"), "estimated_cost_per_person": Decimal("150.00"), "remaining_budget": Decimal("4700.00"), "missing_categories": ["local_transportation"], "estimate_status": "partial", "budget_comparison_status": "provisional"},
        "travel_information": {"weather_summary": "Forecast available.", "transport_options": ["Metro"], "packing_recommendations": ["Comfortable shoes"], "safety_tips": ["Carry identification."], "sources": [{"url": "https://example.test/travel", "information_type": "web_research"}]},
        "validation_result": {"status": "PARTIAL", "is_valid": True, "checks_performed": 2, "checks_passed": 1, "error_count": 0, "warning_count": 1, "issues": [{"severity": "PASS", "category": "travel_request", "code": "ok", "message": "Request valid."}, {"severity": "WARNING", "category": "budget", "code": "partial", "message": "Budget is partial."}]},
    }
    state.update(overrides)
    return state


def test_complete_available_outputs_are_formatted_without_recalculation() -> None:
    plan = generate_final_response(_state())  # type: ignore[arg-type]
    text = plan.markdown_response
    assert "## Trip Overview" in text and "**Destination:** Pune" in text
    assert "### Day 1" in text and text.index("### Day 1") < text.index("### Day 2")
    assert "**Known subtotal:** INR 300.00" in text
    assert "**Final estimated total:** INR 300.00" in text
    assert "Entry fee:** Available in research but unverified" in text
    assert "Entry fee (verified):" not in text
    assert "**Status:** PARTIAL" in text and "Budget is partial." in text
    assert plan.source_urls == ["https://example.test/research", "https://example.test/price", "https://example.test/travel"]


def test_missing_outputs_produce_partial_plan_without_fabrication() -> None:
    plan = generate_final_response({"destination": "Pune", "currency": "INR", "errors": []})  # type: ignore[arg-type]
    assert plan.generation_status == "partial"
    assert "Destination research is unavailable." in plan.markdown_response
    assert "no substitute activities were created" in plan.markdown_response
    assert "Budget breakdown could not be generated." in plan.markdown_response
    assert "Validation report is unavailable." in plan.markdown_response


def test_agent_writes_only_its_output_and_preserves_input() -> None:
    state = _state()
    before = dict(state)
    update = final_response_agent(state)  # type: ignore[arg-type]
    assert set(update) == {"final_plan", "agent_status"}
    assert update["agent_status"] == {"final_response_generator": "completed_partial"}
    assert state == before


def test_failed_validation_is_prominent_and_external_calls_are_not_used() -> None:
    plan = generate_final_response(_state(validation_result={"status": "FAILED", "is_valid": False, "checks_performed": 1, "checks_passed": 0, "error_count": 1, "warning_count": 0, "issues": [{"severity": "ERROR", "category": "budget", "code": "bad", "message": "Budget mismatch."}]}))  # type: ignore[arg-type]
    assert "**Status:** FAILED" in plan.markdown_response
    assert "This travel plan needs attention" in plan.markdown_response
    from backend.agents import final_response_agent as module
    assert not hasattr(module, "get_llm")


@pytest.mark.parametrize("fragment", [
    "## Trip Overview", "**Travellers:** 2", "**Start date:** 2027-02-01",
    "## Destination Overview", "A cultural city.", "## Recommended Attractions",
    "**Location:** Centre", "**Estimated visit duration:** 2 hours",
    "## Day-wise Itinerary", "**Activities:** Morning museum", "**Attractions:** Museum",
    "## Budget Breakdown", "**Accommodation:** INR 100.00", "**Food:** INR 200.00",
    "**Emergency buffer:** INR 0.00", "**Per-person cost:** INR 150.00",
    "**Remaining budget:** INR 4700.00", "## Weather and Travel Information",
    "**Transportation:** Metro", "**Packing recommendations:** Comfortable shoes",
    "## Validation Report", "**Warnings:** 1", "## Important Notes",
    "Prior error is preserved.", "## Sources", "https://example.test/research",
])
def test_all_display_sections_preserve_available_content(fragment: str) -> None:
    assert fragment in generate_final_response(_state()).markdown_response  # type: ignore[arg-type]


@pytest.mark.parametrize("removed, expected", [
    ("destination_research", "Destination research is unavailable."),
    ("itinerary", "Day-wise itinerary is unavailable"),
    ("budget_breakdown", "Budget breakdown could not be generated."),
    ("travel_information", "Weather and travel information is unavailable."),
    ("validation_result", "Validation report is unavailable."),
])
def test_each_missing_agent_output_is_explicit(removed: str, expected: str) -> None:
    state = _state()
    state.pop(removed)
    assert expected in generate_final_response(state).markdown_response  # type: ignore[arg-type]
