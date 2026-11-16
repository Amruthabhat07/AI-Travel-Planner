"""Offline orchestration tests for the four-agent LangGraph workflow."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from backend.graph import workflow as workflow_module
from backend.graph.state import TravelState
from backend.models.schemas import BudgetPriceInput, DestinationResearch, TravelRequest


def _request() -> TravelRequest:
    return TravelRequest(
        origin="Origin City",
        destination="Destination City",
        start_date=date(2027, 6, 1),
        end_date=date(2027, 6, 2),
        duration=2,
        budget=1000,
        currency="INR",
        travelers=2,
        preferences=["history"],
    )


def _research() -> dict[str, Any]:
    return DestinationResearch(
        destination="Destination City",
        overview="Researched overview.",
        attractions=[{"name": "Museum", "description": "Researched attraction."}],
        research_sources=["https://example.test/destination"],
    ).model_dump(mode="python")


def _install_successful_agents(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, dict[str, Any]]]) -> None:
    """Install deterministic nodes while retaining actual StateGraph behavior."""

    def research(state: TravelState) -> dict[str, Any]:
        calls.append(("destination_research", dict(state)))
        return {
            "destination_research": _research(),
            "agent_status": {"destination_research": "completed"},
        }

    def itinerary(state: TravelState) -> dict[str, Any]:
        calls.append(("itinerary_planning", dict(state)))
        return {
            "itinerary": [{"day_number": 1, "title": "Explore", "activities": ["Museum"]}],
            "agent_status": {"itinerary_planning": "completed_with_geographic_optimization"},
        }

    def budget(state: TravelState) -> dict[str, Any]:
        calls.append(("budget_planning", dict(state)))
        return {
            "budget_breakdown": {"currency": "INR", "total_estimated_cost": "250.00"},
            "agent_status": {"budget_planning": "completed_partial"},
        }

    def assistant(state: TravelState) -> dict[str, Any]:
        calls.append(("travel_assistant", dict(state)))
        return {
            "travel_information": {"weather_summary": "Forecast available."},
            "agent_status": {"travel_assistant": "completed"},
        }

    def validation(state: TravelState) -> dict[str, Any]:
        calls.append(("validation_agent", dict(state)))
        return {
            "validation_result": {"status": "PASSED", "is_valid": True},
            "agent_status": {"validation_agent": "completed"},
        }

    monkeypatch.setattr(workflow_module, "destination_research_agent", research)
    monkeypatch.setattr(workflow_module, "itinerary_planning_agent", itinerary)
    monkeypatch.setattr(workflow_module, "budget_planning_agent", budget)
    monkeypatch.setattr(workflow_module, "travel_assistant_agent", assistant)
    monkeypatch.setattr(workflow_module, "validation_agent", validation)


def _install_basic_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fast simple-success functions for isolated routing tests."""
    monkeypatch.setattr(
        workflow_module,
        "destination_research_agent",
        lambda _state: {
            "destination_research": _research(),
            "agent_status": {"destination_research": "completed"},
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "itinerary_planning_agent",
        lambda _state: {
            "itinerary": [{"day_number": 1, "title": "Plan"}],
            "agent_status": {"itinerary_planning": "completed"},
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "budget_planning_agent",
        lambda _state: {
            "budget_breakdown": {"currency": "INR"},
            "agent_status": {"budget_planning": "completed"},
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "travel_assistant_agent",
        lambda _state: {
            "travel_information": {"weather_summary": "Available"},
            "agent_status": {"travel_assistant": "completed"},
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "validation_agent",
        lambda _state: {
            "validation_result": {"status": "PASSED", "is_valid": True},
            "agent_status": {"validation_agent": "completed"},
        },
    )


def test_graph_builds_and_compiles() -> None:
    compiled = workflow_module.build_workflow()

    assert isinstance(compiled, CompiledStateGraph)
    assert compiled.get_graph() is not None


def test_all_agent_nodes_are_registered() -> None:
    nodes = workflow_module.get_workflow().get_graph().nodes

    assert set(workflow_module.AGENT_NODE_NAMES).issubset(nodes)
    assert {"skip_itinerary", "skip_budget"}.issubset(nodes)


def test_successful_agents_run_in_defined_order_and_outputs_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    _install_successful_agents(monkeypatch, calls)
    compiled = workflow_module.build_workflow()

    result = compiled.invoke(workflow_module.create_initial_travel_state(_request()))

    assert [name for name, _ in calls] == [
        "destination_research", "itinerary_planning", "budget_planning", "travel_assistant", "validation_agent"
    ]
    assert calls[1][1]["destination_research"]["destination"] == "Destination City"
    assert calls[2][1]["itinerary"][0]["title"] == "Explore"
    assert calls[3][1]["budget_breakdown"]["total_estimated_cost"] == "250.00"
    assert calls[4][1]["travel_information"]["weather_summary"] == "Forecast available."
    assert result["itinerary"][0]["title"] == "Explore"
    assert result["budget_breakdown"]["total_estimated_cost"] == "250.00"
    assert result["travel_information"]["weather_summary"] == "Forecast available."
    assert result["destination_research"]["overview"] == "Researched overview."
    assert result["agent_status"] == {
        "destination_research": "completed",
        "itinerary_planning": "completed_with_geographic_optimization",
        "budget_planning": "completed_partial",
        "travel_assistant": "completed",
        "validation_agent": "completed",
        "final_response_generator": "completed_partial",
    }
    assert result["validation_result"]["status"] == "PASSED"


def test_research_failure_skips_itinerary_but_continues_budget_and_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        workflow_module,
        "destination_research_agent",
        lambda _state: {
            "agent_status": {"destination_research": "failed"},
            "errors": ["mock research failure"],
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "itinerary_planning_agent",
        MagicMock(side_effect=AssertionError("itinerary must be skipped")),
    )
    monkeypatch.setattr(
        workflow_module,
        "budget_planning_agent",
        lambda _state: calls.append("budget") or {"agent_status": {"budget_planning": "completed_partial"}},
    )
    monkeypatch.setattr(
        workflow_module,
        "travel_assistant_agent",
        lambda _state: calls.append("assistant") or {"travel_information": {"packing_recommendations": []}, "agent_status": {"travel_assistant": "completed_with_warnings"}},
    )
    monkeypatch.setattr(
        workflow_module,
        "validation_agent",
        lambda _state: {"validation_result": {"status": "PARTIAL", "is_valid": True}, "agent_status": {"validation_agent": "completed_partial"}},
    )

    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    assert calls == ["budget", "assistant"]
    assert result["agent_status"]["destination_research"] == "failed"
    assert result["agent_status"]["itinerary_planning"] == "skipped_missing_destination_research"
    assert result["agent_status"]["budget_planning"] == "completed_partial"
    assert "mock research failure" in result["errors"]
    assert any("Itinerary planning skipped" in error for error in result["errors"])
    assert "itinerary" not in result


def test_research_with_empty_attractions_skips_itinerary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow_module,
        "destination_research_agent",
        lambda _state: {
            "destination_research": {"destination": "Destination City", "attractions": []},
            "agent_status": {"destination_research": "completed"},
        },
    )
    itinerary = MagicMock()
    monkeypatch.setattr(workflow_module, "itinerary_planning_agent", itinerary)
    _install_downstream_success_agents(monkeypatch)

    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    itinerary.assert_not_called()
    assert result["agent_status"]["itinerary_planning"] == "skipped_missing_destination_research"


def _install_downstream_success_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        workflow_module,
        "budget_planning_agent",
        lambda _state: {"agent_status": {"budget_planning": "completed_partial"}},
    )
    monkeypatch.setattr(
        workflow_module,
        "travel_assistant_agent",
        lambda _state: {"agent_status": {"travel_assistant": "completed_with_warnings"}},
    )
    monkeypatch.setattr(
        workflow_module,
        "validation_agent",
        lambda _state: {"validation_result": {"status": "PARTIAL", "is_valid": True}, "agent_status": {"validation_agent": "completed_partial"}},
    )


def test_itinerary_failure_preserves_research_and_continues_later_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        workflow_module,
        "destination_research_agent",
        lambda _state: {
            "destination_research": _research(),
            "agent_status": {"destination_research": "completed"},
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "itinerary_planning_agent",
        lambda _state: {"agent_status": {"itinerary_planning": "failed"}, "errors": ["itinerary failed"]},
    )
    monkeypatch.setattr(
        workflow_module,
        "budget_planning_agent",
        lambda state: seen.append(dict(state)) or {"agent_status": {"budget_planning": "completed_partial"}},
    )
    monkeypatch.setattr(
        workflow_module,
        "travel_assistant_agent",
        lambda state: seen.append(dict(state)) or {"travel_information": {"packing_recommendations": []}, "agent_status": {"travel_assistant": "completed_with_warnings"}},
    )
    monkeypatch.setattr(
        workflow_module,
        "validation_agent",
        lambda _state: {"validation_result": {"status": "PARTIAL", "is_valid": True}, "agent_status": {"validation_agent": "completed_partial"}},
    )

    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    assert len(seen) == 2
    assert all(item["destination_research"]["destination"] == "Destination City" for item in seen)
    assert result["agent_status"]["itinerary_planning"] == "failed"
    assert "itinerary" not in result
    assert result["destination_research"]["destination"] == "Destination City"
    assert "itinerary failed" in result["errors"]
    assert result["travel_information"]["packing_recommendations"] == []


def test_budget_failure_preserves_upstream_outputs_and_runs_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_basic_agents(monkeypatch)
    monkeypatch.setattr(
        workflow_module,
        "budget_planning_agent",
        lambda _state: {"agent_status": {"budget_planning": "failed"}, "errors": ["budget unavailable"]},
    )
    assistant_seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        workflow_module,
        "travel_assistant_agent",
        lambda state: assistant_seen.append(dict(state)) or {"agent_status": {"travel_assistant": "completed_with_warnings"}},
    )
    monkeypatch.setattr(
        workflow_module,
        "validation_agent",
        lambda _state: {"validation_result": {"status": "PARTIAL", "is_valid": True}, "agent_status": {"validation_agent": "completed_partial"}},
    )

    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    assert assistant_seen[0]["destination_research"]
    assert assistant_seen[0]["itinerary"]
    assert "budget_breakdown" not in assistant_seen[0]
    assert result["agent_status"]["budget_planning"] == "failed"
    assert "budget unavailable" in result["errors"]
    assert result["itinerary"]


def test_travel_assistant_exception_preserves_all_prior_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_basic_agents(monkeypatch)
    monkeypatch.setattr(
        workflow_module,
        "travel_assistant_agent",
        MagicMock(side_effect=RuntimeError("assistant crashed")),
    )

    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    assert result["agent_status"]["travel_assistant"] == "failed"
    assert result["destination_research"]
    assert result["itinerary"]
    assert result["budget_breakdown"]
    assert any("assistant crashed" in error for error in result["errors"])


def test_validation_runs_after_travel_assistant_and_preserves_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_agents(monkeypatch, calls := [])
    state_at_validation: list[dict[str, Any]] = []

    def validation(state: TravelState) -> dict[str, Any]:
        state_at_validation.append(dict(state))
        return {
            "validation_result": {"status": "PASSED_WITH_WARNINGS", "is_valid": True},
            "agent_status": {"validation_agent": "completed_with_warnings"},
            "errors": ["validator note"],
        }

    monkeypatch.setattr(workflow_module, "validation_agent", validation)
    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    assert [name for name, _ in calls][-1] == "travel_assistant"
    assert state_at_validation[0]["travel_information"]["weather_summary"] == "Forecast available."
    assert result["validation_result"]["status"] == "PASSED_WITH_WARNINGS"
    assert result["errors"] == ["validator note"]
    assert result["destination_research"]
    assert result["itinerary"]
    assert result["budget_breakdown"]
    assert result["travel_information"]


def test_validation_agent_exception_does_not_discard_travel_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_basic_agents(monkeypatch)
    monkeypatch.setattr(
        workflow_module,
        "validation_agent",
        MagicMock(side_effect=RuntimeError("validator crashed")),
    )

    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    assert result["agent_status"]["validation_agent"] == "failed"
    assert any("validator crashed" in error for error in result["errors"])
    assert result["destination_research"]
    assert result["itinerary"]
    assert result["budget_breakdown"]
    assert result["travel_information"]


def test_missing_budget_prerequisites_skip_budget_and_preserve_partial_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_basic_agents(monkeypatch)
    budget = MagicMock(side_effect=AssertionError("budget node must be skipped"))
    monkeypatch.setattr(workflow_module, "budget_planning_agent", budget)
    assistant_seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        workflow_module,
        "travel_assistant_agent",
        lambda state: assistant_seen.append(dict(state)) or {"agent_status": {"travel_assistant": "completed_with_warnings"}},
    )
    incomplete = {"destination": "Destination City", "travelers": 2}

    result = workflow_module.build_workflow().invoke(incomplete)

    budget.assert_not_called()
    assert result["agent_status"]["budget_planning"] == "skipped_missing_requirements"
    assert result["agent_status"]["travel_assistant"] == "completed_with_warnings"
    assert "Budget planning skipped" in result["errors"][-1]
    assert assistant_seen


def test_research_agent_exception_is_recorded_and_routes_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow_module,
        "destination_research_agent",
        MagicMock(side_effect=RuntimeError("research node crashed")),
    )
    itinerary = MagicMock(side_effect=AssertionError("itinerary must skip"))
    monkeypatch.setattr(workflow_module, "itinerary_planning_agent", itinerary)
    _install_downstream_success_agents(monkeypatch)

    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    assert result["agent_status"]["destination_research"] == "failed"
    assert result["agent_status"]["itinerary_planning"] == "skipped_missing_destination_research"
    assert any("research node crashed" in error for error in result["errors"])


def test_error_reducer_accumulates_errors_from_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_basic_agents(monkeypatch)
    monkeypatch.setattr(
        workflow_module,
        "itinerary_planning_agent",
        lambda _state: {"agent_status": {"itinerary_planning": "failed"}, "errors": ["itinerary error"]},
    )
    monkeypatch.setattr(
        workflow_module,
        "budget_planning_agent",
        lambda _state: {"agent_status": {"budget_planning": "failed"}, "errors": ["budget error"]},
    )
    monkeypatch.setattr(
        workflow_module,
        "travel_assistant_agent",
        lambda _state: {"agent_status": {"travel_assistant": "completed_with_warnings"}},
    )
    monkeypatch.setattr(
        workflow_module,
        "validation_agent",
        lambda _state: {"validation_result": {"status": "FAILED", "is_valid": False}, "agent_status": {"validation_agent": "failed"}},
    )

    result = workflow_module.build_workflow().invoke(
        workflow_module.create_initial_travel_state(_request())
    )

    assert result["errors"] == ["itinerary error", "budget error"]


def test_unrelated_state_fields_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_basic_agents(monkeypatch)
    initial = {
        **workflow_module.create_initial_travel_state(_request()),
        "travel_tips": ["preserve this"],
        "daily_schedule": {"day_1": ["existing"]},
    }

    result = workflow_module.build_workflow().invoke(initial)

    assert result["travel_tips"] == ["preserve this"]
    assert result["daily_schedule"] == {"day_1": ["existing"]}


def test_invocation_accepts_request_and_explicit_budget_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_basic_agents(monkeypatch)
    price = BudgetPriceInput(
        category="food",
        name="User food input",
        unit_price="12.50",
        currency="INR",
        pricing_unit="per_person_per_day",
        price_type="user_provided",
        source_name="User",
    )
    result = workflow_module.invoke_travel_workflow(
        _request(),
        user_request="Plan this trip",
        budget_inputs=[price],
        emergency_buffer_percent="5.5",
        workflow=workflow_module.build_workflow(),
    )

    assert result["destination"] == "Destination City"
    assert result["user_request"] == "Plan this trip"
    assert result["budget_inputs"][0]["name"] == "User food input"
    assert result["emergency_buffer_percent"] == 5.5


def test_invocation_rejects_invalid_request_data() -> None:
    with pytest.raises(ValidationError):
        workflow_module.invoke_travel_workflow({"destination": ""})


def test_invocation_rejects_invalid_budget_inputs_and_buffer() -> None:
    with pytest.raises(ValidationError):
        workflow_module.invoke_travel_workflow(
            _request(), budget_inputs=[{"category": "invalid"}]
        )
    with pytest.raises(ValueError, match="between 0 and 100"):
        workflow_module.invoke_travel_workflow(_request(), emergency_buffer_percent=101)


def test_import_and_graph_construction_do_not_call_external_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for agent_name in (
        "destination_research_agent",
        "itinerary_planning_agent",
        "budget_planning_agent",
        "travel_assistant_agent",
        "validation_agent",
    ):
        monkeypatch.setattr(workflow_module, agent_name, MagicMock(side_effect=AssertionError("agent invoked during build")))

    compiled = workflow_module.build_workflow()

    assert compiled.get_graph() is not None
    for agent_name in (
        "destination_research_agent",
        "itinerary_planning_agent",
        "budget_planning_agent",
        "travel_assistant_agent",
        "validation_agent",
    ):
        getattr(workflow_module, agent_name).assert_not_called()


def test_mermaid_inspection_exposes_nodes_and_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_basic_agents(monkeypatch)
    compiled = workflow_module.build_workflow()

    diagram = workflow_module.get_workflow_mermaid(workflow=compiled)

    for node in (*workflow_module.AGENT_NODE_NAMES, "skip_itinerary", "skip_budget"):
        assert node in diagram
