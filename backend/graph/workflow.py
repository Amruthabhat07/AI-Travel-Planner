"""LangGraph orchestration for the four existing travel-planning agents.

Importing this module defines functions only; it performs no LLM, Tavily, or
weather requests. A compiled graph is created lazily and cached on first use.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from backend.agents.budget_planning_agent import budget_planning_agent
from backend.agents.destination_research_agent import destination_research_agent
from backend.agents.itinerary_planning_agent import itinerary_planning_agent
from backend.agents.travel_assistant_agent import travel_assistant_agent
from backend.graph.state import TravelState, create_initial_travel_state
from backend.models.schemas import BudgetPriceInput, DestinationResearch, TravelRequest

logger = logging.getLogger(__name__)

AGENT_NODE_NAMES = (
    "destination_research",
    "itinerary_planning",
    "budget_planning",
    "travel_assistant",
)

AgentCallable = Callable[[TravelState], Mapping[str, Any]]


def _status(state: Mapping[str, Any], agent_key: str) -> str | None:
    """Read an agent status safely from reducer-backed state."""
    statuses = state.get("agent_status")
    if not isinstance(statuses, Mapping):
        return None
    value = statuses.get(agent_key)
    return value if isinstance(value, str) else None


def _failure_update(agent_key: str, message: str) -> dict[str, Any]:
    """Create a failure update preserving shared error/status reducers."""
    return {
        "agent_status": {agent_key: "failed"},
        "errors": [message],
    }


def _run_agent(agent_key: str, agent: AgentCallable, state: TravelState) -> dict[str, Any]:
    """Invoke an existing agent safely and ensure it returns a partial mapping."""
    try:
        update = agent(state)
        if not isinstance(update, Mapping):
            raise TypeError(f"{agent_key} agent did not return a mapping state update.")
        return dict(update)
    except Exception as exc:  # noqa: BLE001 - prevent a node exception stopping partial workflow results.
        logger.exception("Workflow node %s raised an exception", agent_key)
        return _failure_update(
            agent_key,
            f"{agent_key} agent raised an exception ({type(exc).__name__}): {exc}",
        )


def _research_is_usable(state: Mapping[str, Any]) -> bool:
    """Check both the upstream status and the research payload before itinerary planning."""
    status = _status(state, "destination_research")
    if status == "failed" or (status is not None and status.startswith("skipped")):
        return False
    try:
        research = DestinationResearch.model_validate(state.get("destination_research"))
    except (ValidationError, TypeError, ValueError):
        return False
    return bool(research.attractions)


def _itinerary_route(state: TravelState) -> str:
    """Route to itinerary planning only after usable destination research."""
    return "itinerary_planning" if _research_is_usable(state) else "skip_itinerary"


def _budget_requirements_are_sufficient(state: Mapping[str, Any]) -> bool:
    """Apply the budget agent's actual required request/duration checks."""
    # The budget agent requires an explicit traveler count even though
    # TravelRequest itself has a schema default for direct request validation.
    if "travelers" not in state or state.get("travelers") is None:
        return False
    request_fields = (
        "origin",
        "destination",
        "start_date",
        "end_date",
        "duration",
        "budget",
        "currency",
        "travelers",
        "preferences",
        "travel_style",
        "additional_requirements",
    )
    try:
        request = TravelRequest.model_validate(
            {key: state[key] for key in request_fields if key in state}
        )
    except (ValidationError, TypeError, ValueError):
        return False
    duration = request.duration
    if duration is None and request.start_date is not None and request.end_date is not None:
        duration = (request.end_date - request.start_date).days + 1
    return bool(
        request.destination.strip()
        and request.travelers >= 1
        and duration is not None
        and duration > 0
    )


def _budget_route(state: TravelState) -> str:
    """Avoid invoking budget calculations until their required fields exist."""
    return "budget_planning" if _budget_requirements_are_sufficient(state) else "skip_budget"


def _skip_itinerary_node(state: TravelState) -> dict[str, Any]:
    """Mark itinerary as skipped when no valid research/attractions exist."""
    status = _status(state, "destination_research")
    reason = (
        "Destination research failed."
        if status == "failed"
        else "Usable destination research with attractions is unavailable."
    )
    return {
        "agent_status": {"itinerary_planning": "skipped_missing_destination_research"},
        "errors": [f"Itinerary planning skipped: {reason}"],
    }


def _skip_budget_node(state: TravelState) -> dict[str, Any]:
    """Mark budget as skipped when request fields required for calculation are absent."""
    missing = [
        key
        for key in ("destination", "travelers")
        if state.get(key) is None or (key == "destination" and not state.get(key))
    ]
    if state.get("duration") is None and not (
        state.get("start_date") is not None and state.get("end_date") is not None
    ):
        missing.append("duration or complete travel dates")
    return {
        "agent_status": {"budget_planning": "skipped_missing_requirements"},
        "errors": [
            "Budget planning skipped: required travel data is missing or invalid. "
            f"Missing: {', '.join(missing) or 'positive duration or valid date range'}."
        ],
    }


def build_workflow() -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build and compile a fresh sequential workflow with safe failure routes.

    Node function names match the corresponding existing agent implementations.
    After research, itinerary planning runs only when a validated research payload
    with attractions exists. Budget planning is independently gated by its
    required request data. The Travel Assistant is always allowed to attempt a
    partial result so it can return general guidance when optional outputs fail.
    """
    graph = StateGraph(TravelState)

    graph.add_node(
        "destination_research",
        lambda state: _run_agent("destination_research", destination_research_agent, state),
    )
    graph.add_node(
        "itinerary_planning",
        lambda state: _run_agent("itinerary_planning", itinerary_planning_agent, state),
    )
    graph.add_node(
        "skip_itinerary",
        _skip_itinerary_node,
    )
    graph.add_node(
        "budget_planning",
        lambda state: _run_agent("budget_planning", budget_planning_agent, state),
    )
    graph.add_node("skip_budget", _skip_budget_node)
    graph.add_node(
        "travel_assistant",
        lambda state: _run_agent("travel_assistant", travel_assistant_agent, state),
    )

    graph.add_edge(START, "destination_research")
    graph.add_conditional_edges(
        "destination_research",
        _itinerary_route,
        {
            "itinerary_planning": "itinerary_planning",
            "skip_itinerary": "skip_itinerary",
        },
    )
    graph.add_conditional_edges(
        "itinerary_planning",
        _budget_route,
        {
            "budget_planning": "budget_planning",
            "skip_budget": "skip_budget",
        },
    )
    graph.add_conditional_edges(
        "skip_itinerary",
        _budget_route,
        {
            "budget_planning": "budget_planning",
            "skip_budget": "skip_budget",
        },
    )
    graph.add_edge("budget_planning", "travel_assistant")
    graph.add_edge("skip_budget", "travel_assistant")
    graph.add_edge("travel_assistant", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_workflow() -> CompiledStateGraph[Any, Any, Any, Any]:
    """Return the cached compiled travel workflow, compiling lazily on demand."""
    return build_workflow()


def invoke_travel_workflow(
    request: TravelRequest | Mapping[str, Any],
    *,
    user_request: str | None = None,
    budget_inputs: list[BudgetPriceInput | Mapping[str, Any]] | None = None,
    emergency_buffer_percent: Decimal | float | int | str | None = None,
    workflow: CompiledStateGraph[Any, Any, Any, Any] | None = None,
) -> TravelState:
    """Validate input, initialize shared state, and invoke the compiled workflow.

    Args:
        request: Existing validated request model or a mapping accepted by it.
        user_request: Optional original natural-language request.
        budget_inputs: Explicit source-aware price inputs for the Python budget agent.
        emergency_buffer_percent: Optional explicit buffer percentage (default is zero).
        workflow: Optional compiled graph override, useful for isolated tests.

    Returns:
        The final partial-updated shared state after all applicable workflow nodes.

    Raises:
        pydantic.ValidationError: When the input request or price inputs are invalid.
        ValueError: When emergency-buffer configuration is invalid.
    """
    validated_request = (
        request
        if isinstance(request, TravelRequest)
        else TravelRequest.model_validate(request)
    )
    state = create_initial_travel_state(validated_request, user_request=user_request)

    if budget_inputs is not None:
        state["budget_inputs"] = [
            item.model_dump(mode="python", exclude_none=True)
            if isinstance(item, BudgetPriceInput)
            else BudgetPriceInput.model_validate(item).model_dump(mode="python", exclude_none=True)
            for item in budget_inputs
        ]

    if emergency_buffer_percent is not None:
        try:
            value = Decimal(str(emergency_buffer_percent))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError("Emergency buffer percentage must be a finite number.") from exc
        if not value.is_finite() or value < 0 or value > 100:
            raise ValueError("Emergency buffer percentage must be between 0 and 100.")
        state["emergency_buffer_percent"] = float(value)

    compiled = workflow if workflow is not None else get_workflow()
    result = compiled.invoke(state)
    if not isinstance(result, dict):
        raise TypeError("Compiled travel workflow returned a non-dictionary state.")
    return result


def get_workflow_mermaid(*, workflow: CompiledStateGraph[Any, Any, Any, Any] | None = None) -> str:
    """Return a Mermaid diagram of the compiled workflow for inspection."""
    compiled = workflow if workflow is not None else get_workflow()
    return compiled.get_graph().draw_mermaid()
