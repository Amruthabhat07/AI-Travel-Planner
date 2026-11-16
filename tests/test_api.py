"""Offline API tests; the LangGraph workflow is always mocked."""

from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import MagicMock

os.environ.setdefault("GOOGLE_API_KEY", "api-test-google-key")
os.environ.setdefault("TAVILY_API_KEY", "api-test-tavily-key")

from fastapi.testclient import TestClient

from backend.main import create_app


def _payload() -> dict[str, object]:
    return {
        "destination": "Pune", "start_date": "2027-02-01", "end_date": "2027-02-02",
        "duration": 2, "travelers": 2, "budget": 1000, "currency": "INR",
        "emergency_buffer_percent": 5,
    }


def _result(*, partial: bool = False) -> dict[str, object]:
    budget = {
        "currency": "INR", "known_subtotal": Decimal("250.00"), "emergency_buffer": Decimal("0.00"),
        "total_estimated_cost": Decimal("250.00"), "estimate_status": "partial" if partial else "complete",
        "budget_comparison_status": "provisional" if partial else "within_limit",
        "missing_categories": ["food"] if partial else [],
    }
    final_plan = {
        "destination": "Pune", "currency": "INR", "budget_breakdown": budget,
        "generation_status": "partial" if partial else "complete", "markdown_response": "# Pune plan\n",
        "missing_information": ["Food pricing is unavailable."] if partial else [],
        "source_urls": ["https://example.test/source"],
    }
    return {
        "final_plan": final_plan, "budget_breakdown": budget, "itinerary": [],
        "agent_status": {"final_response_generator": "completed_partial" if partial else "completed"},
        "errors": ["Budget is partial."] if partial else [],
    }


def _client() -> TestClient:
    return TestClient(create_app())


def test_root_returns_api_information() -> None:
    response = _client().get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "AI Travel Planner API"
    assert response.json()["docs"] == "/docs"


def test_health_is_operational_without_workflow_execution() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json()["operational"] is True


def test_status_does_not_invoke_workflow(monkeypatch) -> None:
    workflow = MagicMock(side_effect=AssertionError("workflow must not run"))
    monkeypatch.setattr("backend.api.travel.invoke_travel_workflow", workflow)
    response = _client().get("/api/v1/travel/status")
    assert response.status_code == 200
    assert response.json()["workflow_available"] is True
    workflow.assert_not_called()


def test_plan_invokes_workflow_with_validated_request(monkeypatch) -> None:
    workflow = MagicMock(return_value=_result())
    monkeypatch.setattr("backend.api.travel.invoke_travel_workflow", workflow)
    response = _client().post("/api/v1/travel/plan", json=_payload())
    assert response.status_code == 200
    request = workflow.call_args.args[0]
    assert request.destination == "Pune"
    assert workflow.call_args.kwargs["emergency_buffer_percent"] == 5.0


def test_plan_serializes_final_markdown_and_decimal_budget(monkeypatch) -> None:
    monkeypatch.setattr("backend.api.travel.invoke_travel_workflow", MagicMock(return_value=_result()))
    body = _client().post("/api/v1/travel/plan", json=_payload()).json()
    assert body["markdown_response"] == "# Pune plan\n"
    assert body["budget_breakdown"]["total_estimated_cost"] in (250, 250.0, "250.00")


def test_invalid_request_returns_422(monkeypatch) -> None:
    workflow = MagicMock()
    monkeypatch.setattr("backend.api.travel.invoke_travel_workflow", workflow)
    response = _client().post("/api/v1/travel/plan", json={"destination": ""})
    assert response.status_code == 422
    workflow.assert_not_called()


def test_workflow_exception_hides_internal_details(monkeypatch) -> None:
    monkeypatch.setattr("backend.api.travel.invoke_travel_workflow", MagicMock(side_effect=RuntimeError("secret traceback value")))
    response = _client().post("/api/v1/travel/plan", json=_payload())
    assert response.status_code == 500
    assert "secret traceback" not in response.text
    assert response.json()["detail"] == "Travel planning could not be completed."


def test_partial_result_is_returned_with_available_data(monkeypatch) -> None:
    monkeypatch.setattr("backend.api.travel.invoke_travel_workflow", MagicMock(return_value=_result(partial=True)))
    response = _client().post("/api/v1/travel/plan", json=_payload())
    assert response.status_code == 200
    assert response.json()["generation_status"] == "partial"
    assert response.json()["errors"] == ["Budget is partial."]


def test_missing_final_response_returns_safe_gateway_error(monkeypatch) -> None:
    monkeypatch.setattr("backend.api.travel.invoke_travel_workflow", MagicMock(return_value={"errors": ["final agent failed"]}))
    response = _client().post("/api/v1/travel/plan", json=_payload())
    assert response.status_code == 502
    assert response.json()["detail"] == "Travel planning completed without a final response."


def test_workflow_value_error_is_reported_as_client_error(monkeypatch) -> None:
    monkeypatch.setattr("backend.api.travel.invoke_travel_workflow", MagicMock(side_effect=ValueError("invalid buffer")))
    response = _client().post("/api/v1/travel/plan", json=_payload())
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid buffer"
