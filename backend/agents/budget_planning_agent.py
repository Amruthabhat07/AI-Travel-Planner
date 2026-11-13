"""Deterministic, Python-only travel budget calculations for shared state."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from pydantic import ValidationError

from backend.graph.state import TravelState, model_to_state_update
from backend.models.schemas import (
    BudgetBreakdown,
    BudgetCategory,
    BudgetLineItem,
    BudgetPriceInput,
    DestinationResearch,
    TravelRequest,
)

logger = logging.getLogger(__name__)
_AGENT_KEY = "budget_planning"
_CENT = Decimal("0.01")
_ZERO = Decimal("0")
_BUDGET_CATEGORIES: tuple[BudgetCategory, ...] = (
    "accommodation",
    "food",
    "local_transportation",
    "intercity_transportation",
    "activity",
    "other_expenses",
)


def _status_update(status: str, error: str | None = None) -> dict[str, Any]:
    """Return a reducer-compatible status/error update, without budget data."""
    update: dict[str, Any] = {"agent_status": {_AGENT_KEY: status}}
    if error:
        update["errors"] = [error]
    return update


def _decimal(value: Any, label: str) -> Decimal:
    """Convert via string form and reject non-finite numeric values."""
    try:
        converted = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a valid number.") from exc
    if not converted.is_finite():
        raise ValueError(f"{label} must be finite; NaN and infinity are not allowed.")
    return converted


def _round_money(value: Decimal) -> Decimal:
    """Round final monetary values to two places using half-up rounding."""
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def _request_from_state(state: Mapping[str, Any]) -> tuple[TravelRequest, int]:
    """Build and validate requirements, deriving duration only from two dates."""
    if "travelers" not in state or state.get("travelers") is None:
        raise ValueError("A number of travelers is required for budget calculation.")
    if isinstance(state.get("travelers"), bool):
        raise ValueError("Number of travelers must be a positive whole number.")

    data = {
        key: state[key]
        for key in (
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
        if key in state
    }
    try:
        request = TravelRequest.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Travel requirements are invalid: {exc}") from exc

    duration = request.duration
    if duration is None and request.start_date is not None and request.end_date is not None:
        duration = (request.end_date - request.start_date).days + 1
    if duration is None or duration <= 0:
        raise ValueError("A positive trip duration or a valid date range is required.")
    if request.travelers < 1:
        raise ValueError("Number of travelers must be at least one.")
    return request, duration


def _price_inputs(state: Mapping[str, Any], *, travelers: int) -> list[BudgetPriceInput]:
    """Validate explicitly supplied price inputs; do not infer them from prose."""
    raw_inputs = state.get("budget_inputs", [])
    if raw_inputs is None:
        raw_inputs = []
    if not isinstance(raw_inputs, list):
        raise ValueError("budget_inputs must be a list of explicit price records.")
    try:
        inputs = [BudgetPriceInput.model_validate(item) for item in raw_inputs]
    except ValidationError as exc:
        raise ValueError(f"Budget price input is invalid: {exc}") from exc

    # Destination research costs are accepted only when the attraction record
    # carries explicit unit, currency, source, type, and verification metadata.
    raw_research = state.get("destination_research")
    if raw_research is not None:
        try:
            research = DestinationResearch.model_validate(raw_research)
        except ValidationError as exc:
            raise ValueError(f"Destination research in TravelState is invalid: {exc}") from exc
        for attraction in research.attractions:
            if (
                attraction.estimated_cost is None
                or attraction.estimated_cost_currency is None
                or attraction.estimated_cost_pricing_unit is None
                or attraction.estimated_cost_price_type is None
                or attraction.estimated_cost_verification_status is None
            ):
                # Legacy research costs without complete provenance are not
                # trusted as prices; their source remains in destination research.
                continue
            inputs.append(
                BudgetPriceInput(
                    category="activity",
                    name=f"Attraction ticket: {attraction.name}",
                    unit_price=Decimal(str(attraction.estimated_cost)),
                    currency=attraction.estimated_cost_currency,
                    pricing_unit=attraction.estimated_cost_pricing_unit,
                    applicable_travelers=Decimal(travelers),
                    visits=Decimal("1"),
                    source_name=attraction.estimated_cost_source_name,
                    source_url=attraction.estimated_cost_source_url,
                    price_type=attraction.estimated_cost_price_type,
                    verification_status=attraction.estimated_cost_verification_status,
                )
            )

    names: set[tuple[str, str]] = set()
    for item in inputs:
        identity = (item.category, item.name.casefold())
        if identity in names:
            raise ValueError(
                f"Duplicate budget input {item.name!r} in category {item.category!r}; "
                "use one record with the correct quantity to avoid double counting."
            )
        names.add(identity)
    return inputs


def _price_exclusion_reason(item: BudgetPriceInput) -> str | None:
    """Return why a supplied price cannot be trusted for arithmetic."""
    if item.price_type == "published":
        if not (item.source_name or item.source_url):
            return "Published price is missing its source name and URL."
        if item.verification_status != "verified":
            return "Published price is not verified."
    if item.price_type == "estimate" and item.verification_status != "approved":
        return "Estimate has not been manually approved."
    return None


def _quantity_for(
    item: BudgetPriceInput, *, travelers: int, duration: int
) -> Decimal:
    """Calculate an auditable multiplier according to the declared price unit."""
    unit = item.pricing_unit
    if item.category == "accommodation":
        if unit != "per_room_per_night":
            raise ValueError(f"Accommodation price {item.name!r} must use per_room_per_night.")
        if item.rooms is None or item.nights is None:
            raise ValueError(
                f"Accommodation price {item.name!r} requires explicit rooms and nights."
            )
        return _decimal(item.rooms, "Room quantity") * _decimal(item.nights, "Night quantity")

    if item.category == "food":
        if unit != "per_person_per_day":
            raise ValueError(f"Food price {item.name!r} must use per_person_per_day.")
        return Decimal(travelers) * Decimal(duration)

    if item.category == "local_transportation":
        if unit == "per_day":
            return Decimal(duration)
        if unit == "per_trip":
            if item.trips is None:
                raise ValueError(f"Local transport price {item.name!r} requires an explicit trip count.")
            return _decimal(item.trips, "Trip quantity")
        if unit == "per_vehicle":
            if item.vehicles is None:
                raise ValueError(f"Local transport price {item.name!r} requires an explicit vehicle count.")
            return _decimal(item.vehicles, "Vehicle quantity")
        if unit == "per_vehicle_trip":
            if item.vehicles is None or item.trips is None:
                raise ValueError(
                    f"Local transport price {item.name!r} requires explicit vehicles and trips."
                )
            return _decimal(item.vehicles, "Vehicle quantity") * _decimal(item.trips, "Trip quantity")
        raise ValueError(f"Unsupported local transportation unit {unit!r} for {item.name!r}.")

    if item.category == "intercity_transportation":
        if unit == "per_person_ticket":
            return Decimal(travelers)
        if unit == "per_trip":
            if item.trips is None:
                raise ValueError(f"Intercity price {item.name!r} requires an explicit trip count.")
            return _decimal(item.trips, "Trip quantity")
        if unit == "per_vehicle":
            if item.vehicles is None:
                raise ValueError(f"Intercity price {item.name!r} requires an explicit vehicle count.")
            return _decimal(item.vehicles, "Vehicle quantity")
        if unit == "per_vehicle_trip":
            if item.vehicles is None or item.trips is None:
                raise ValueError(
                    f"Intercity price {item.name!r} requires explicit vehicles and trips."
                )
            return _decimal(item.vehicles, "Vehicle quantity") * _decimal(item.trips, "Trip quantity")
        raise ValueError(f"Unsupported intercity transportation unit {unit!r} for {item.name!r}.")

    if item.category == "activity":
        if unit != "per_person_ticket":
            raise ValueError(f"Activity price {item.name!r} must use per_person_ticket.")
        if item.visits is None:
            raise ValueError(f"Activity price {item.name!r} requires an explicit visit count.")
        applicable_travelers = (
            _decimal(item.applicable_travelers, "Applicable traveler count")
            if item.applicable_travelers is not None
            else Decimal(travelers)
        )
        return applicable_travelers * _decimal(item.visits, "Visit quantity")

    if item.category == "other_expenses":
        if unit == "flat":
            return Decimal(1)
        if unit == "per_item":
            if item.quantity is None:
                raise ValueError(f"Other expense {item.name!r} requires an explicit item quantity.")
            return _decimal(item.quantity, "Expense quantity")
        if unit == "per_person_ticket":
            return Decimal(travelers)
        if unit == "per_person_per_day":
            return Decimal(travelers) * Decimal(duration)
        raise ValueError(f"Unsupported other-expense unit {unit!r} for {item.name!r}.")

    raise ValueError(f"Unsupported budget category {item.category!r}.")


def calculate_budget(state: TravelState) -> BudgetBreakdown:
    """Calculate known expenses with Decimal and preserve price provenance.

    Missing categories remain ``None`` in component fields and are listed as
    missing. Approved estimates are included but mark the output as assumption-
    based. No currency conversion or AI-generated price extraction is performed.
    """
    request, duration = _request_from_state(state)
    price_inputs = _price_inputs(state, travelers=request.travelers)
    currency = request.currency.upper()
    travelers = request.travelers

    buffer_raw = _decimal(state.get("emergency_buffer_percent", 0), "Emergency buffer percentage")
    if buffer_raw < _ZERO or buffer_raw > Decimal("100"):
        raise ValueError("Emergency buffer percentage must be between 0 and 100.")

    known_by_category: dict[BudgetCategory, Decimal] = {category: _ZERO for category in _BUDGET_CATEGORIES}
    categories_with_prices: set[BudgetCategory] = set()
    categories_with_unresolved_prices: set[BudgetCategory] = set()
    line_items: list[BudgetLineItem] = []
    estimates_used = False

    for item in price_inputs:
        if item.currency.upper() != currency:
            raise ValueError(
                f"Currency mismatch for {item.name!r}: {item.currency} cannot be added to {currency}. "
                "Supply a valid, sourced conversion explicitly before calculation."
            )
        quantity = _quantity_for(item, travelers=travelers, duration=duration)
        if quantity < _ZERO:
            raise ValueError(f"Quantity for {item.name!r} cannot be negative.")
        unit_price = _decimal(item.unit_price, f"Price for {item.name}")
        if unit_price < _ZERO:
            raise ValueError(f"Price for {item.name!r} cannot be negative.")

        exclusion_reason = _price_exclusion_reason(item)
        if exclusion_reason:
            categories_with_unresolved_prices.add(item.category)
            line_items.append(
                BudgetLineItem(
                    category=item.category,
                    name=item.name,
                    unit_price=unit_price,
                    currency=currency,
                    pricing_unit=item.pricing_unit,
                    quantity=quantity,
                    calculated_amount=None,
                    source_name=item.source_name,
                    source_url=item.source_url,
                    price_type=item.price_type,
                    verification_status=item.verification_status,
                    included_in_total=False,
                    exclusion_reason=exclusion_reason,
                )
            )
            continue

        exact_amount = unit_price * quantity
        rounded_amount = _round_money(exact_amount)
        known_by_category[item.category] += exact_amount
        categories_with_prices.add(item.category)
        estimates_used = estimates_used or item.price_type == "estimate"
        line_items.append(
            BudgetLineItem(
                category=item.category,
                name=item.name,
                unit_price=unit_price,
                currency=currency,
                pricing_unit=item.pricing_unit,
                quantity=quantity,
                calculated_amount=rounded_amount,
                source_name=item.source_name,
                source_url=item.source_url,
                price_type=item.price_type,
                verification_status=item.verification_status,
                included_in_total=True,
            )
        )

    missing_categories = [
        category
        for category in _BUDGET_CATEGORIES
        if category not in categories_with_prices or category in categories_with_unresolved_prices
    ]
    has_known_prices = any(item.included_in_total for item in line_items)
    known_subtotal_raw = sum(known_by_category.values(), _ZERO)
    known_subtotal = _round_money(known_subtotal_raw)
    emergency_buffer_raw = known_subtotal_raw * buffer_raw / Decimal("100")
    emergency_buffer = _round_money(emergency_buffer_raw)
    total_raw = known_subtotal_raw + emergency_buffer_raw
    total_estimated_cost = _round_money(total_raw) if has_known_prices else None
    cost_per_person = (
        _round_money(total_raw / Decimal(travelers)) if has_known_prices and travelers else None
    )

    user_budget = _decimal(request.budget, "User budget") if request.budget is not None else None
    remaining_budget: Decimal | None = None
    utilization: Decimal | None = None
    if user_budget is not None:
        if user_budget <= _ZERO:
            raise ValueError("User budget must be positive to calculate utilization safely.")
        if total_estimated_cost is not None:
            remaining_budget = _round_money(user_budget - total_estimated_cost)
            utilization = (total_estimated_cost / user_budget * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

    if missing_categories:
        estimate_status = "partial_with_assumptions" if estimates_used else "partial"
        comparison_status = "provisional" if user_budget is not None else "not_available"
    else:
        estimate_status = "assumption_based" if estimates_used else "complete"
        if user_budget is None or total_estimated_cost is None:
            comparison_status = "not_available"
        else:
            comparison_status = "within_limit" if total_estimated_cost <= user_budget else "over_budget"

    accommodation = (
        _round_money(known_by_category["accommodation"])
        if "accommodation" in categories_with_prices
        else None
    )
    food = _round_money(known_by_category["food"]) if "food" in categories_with_prices else None
    local_transport = (
        _round_money(known_by_category["local_transportation"])
        if "local_transportation" in categories_with_prices
        else None
    )
    intercity_transport = (
        _round_money(known_by_category["intercity_transportation"])
        if "intercity_transportation" in categories_with_prices
        else None
    )
    transportation = (
        _round_money(
            known_by_category["local_transportation"]
            + known_by_category["intercity_transportation"]
        )
        if local_transport is not None or intercity_transport is not None
        else None
    )
    activities = _round_money(known_by_category["activity"]) if "activity" in categories_with_prices else None
    other = _round_money(known_by_category["other_expenses"]) if "other_expenses" in categories_with_prices else None

    return BudgetBreakdown(
        accommodation_cost=accommodation,
        food_cost=food,
        transportation_cost=transportation,
        activity_cost=activities,
        miscellaneous_cost=other,
        total_estimated_cost=total_estimated_cost,
        currency=currency,
        remaining_budget=remaining_budget,
        local_transportation_cost=local_transport,
        intercity_transportation_cost=intercity_transport,
        line_items=line_items,
        known_subtotal=known_subtotal,
        emergency_buffer=emergency_buffer,
        emergency_buffer_percent=buffer_raw,
        estimated_cost_per_person=cost_per_person,
        user_budget=user_budget,
        budget_utilization_percent=utilization,
        missing_categories=missing_categories,
        estimate_status=estimate_status,
        budget_comparison_status=comparison_status,
    )


def budget_planning_agent(state: TravelState) -> dict[str, Any]:
    """Calculate a budget without invoking Gemini, LangChain, or any LLM.

    The agent reads the flattened validated request fields and explicit
    ``budget_inputs`` from shared state, then returns only ``budget_breakdown``
    plus its reducer-compatible operational status and errors.
    """
    try:
        breakdown = calculate_budget(state)
    except (ValidationError, ValueError, TypeError) as exc:
        logger.info("Budget planning could not complete: %s", exc)
        return _status_update("failed", str(exc))
    except Exception as exc:  # noqa: BLE001 - prevent malformed external state crashing the graph.
        logger.exception("Unexpected budget calculation failure")
        return _status_update("failed", f"Unexpected budget calculation failure ({type(exc).__name__}).")

    if breakdown.estimate_status in {"partial", "partial_with_assumptions"}:
        status = "completed_partial"
    elif breakdown.estimate_status == "assumption_based":
        status = "completed_with_assumptions"
    else:
        status = "completed"
    return {
        "budget_breakdown": model_to_state_update(breakdown),
        "agent_status": {_AGENT_KEY: status},
    }
