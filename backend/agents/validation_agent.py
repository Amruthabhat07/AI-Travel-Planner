"""Deterministic, Python-only structural and mathematical travel validation."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from datetime import timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from backend.graph.state import TravelState, model_to_state_update
from backend.models.schemas import (
    AdviceItem,
    BudgetBreakdown,
    BudgetLineItem,
    DestinationResearch,
    ItineraryDay,
    TravelInformation,
    TravelRequest,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)
_AGENT_KEY = "validation_agent"
_CENT = Decimal("0.01")
_ZERO = Decimal("0")
_SUCCESS_STATUSES = {
    "completed",
    "completed_with_warnings",
    "completed_partial",
    "completed_with_assumptions",
    "completed_limited_geographic_optimization",
    "completed_with_geographic_optimization",
}
_EXACT_TRANSPORT_CLAIM = re.compile(
    r"(?:[$₹€£]\s*\d|\b(?:INR|USD|EUR|GBP)\s*\d|\b\d{1,2}:\d{2}\b|"
    r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?|km|kilometers?|miles?)\b)",
    re.IGNORECASE,
)


def _decimal(value: Any, label: str) -> Decimal:
    """Convert values through their string representation and reject non-finite numbers."""
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{label} is not a valid decimal number.") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite; NaN and infinity are invalid.")
    return result


def _money(value: Decimal) -> Decimal:
    """Apply the Phase 6 two-decimal, ROUND_HALF_UP currency policy."""
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def _plain(value: Any) -> Any:
    """Convert Pydantic models into mappings for common validators."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python", exclude_none=False)
    return value


def _valid_url(value: str) -> bool:
    """Accept only absolute HTTP(S) URLs with a hostname."""
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class _ValidationCollector:
    """Collect structured PASS/WARNING/ERROR issues and check-derived counters."""

    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(
        self,
        category: ValidationCategory,
        severity: ValidationSeverity,
        code: str,
        message: str,
        related_field: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                category=category,
                code=code,
                message=message,
                related_field=related_field,
            )
        )

    def passed(self, category: ValidationCategory, code: str, message: str, field: str | None = None) -> None:
        self.add(category, "PASS", code, message, field)

    def warning(self, category: ValidationCategory, code: str, message: str, field: str | None = None) -> None:
        self.add(category, "WARNING", code, message, field)

    def error(self, category: ValidationCategory, code: str, message: str, field: str | None = None) -> None:
        self.add(category, "ERROR", code, message, field)


def _request_data(state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract flattened TravelRequest fields and normalize ISO date strings."""
    request_fields = (
        "origin", "destination", "start_date", "end_date", "duration", "budget",
        "currency", "travelers", "preferences", "travel_style", "additional_requirements",
    )
    return {key: state[key] for key in request_fields if key in state}


def _validate_request(
    state: Mapping[str, Any], collector: _ValidationCollector
) -> TravelRequest | None:
    """Validate flattened shared request values using the canonical Pydantic model."""
    data = _request_data(state)
    if not data:
        collector.warning(
            "travel_request", "request_missing", "Travel request is unavailable; dependent checks are partial.", "travel_request"
        )
        return None

    if data.get("budget") is not None and "currency" not in data:
        collector.error(
            "travel_request", "currency_missing", "Currency must be explicit when a budget amount is supplied.", "currency"
        )

    try:
        request = TravelRequest.model_validate(data)
    except ValidationError as exc:
        collector.error(
            "travel_request", "request_invalid", f"Travel request failed Pydantic validation: {exc}", "travel_request"
        )
        return None

    collector.passed("travel_request", "request_schema_valid", "Travel request satisfies TravelRequest constraints.", "travel_request")
    if request.destination.strip():
        collector.passed("travel_request", "destination_present", "Destination is present and non-empty.", "destination")
    if request.travelers >= 1:
        collector.passed("travel_request", "travelers_positive", "Traveler count is positive.", "travelers")
    if request.duration is not None:
        collector.passed("travel_request", "duration_positive", "Supplied trip duration is positive.", "duration")
    if request.start_date is not None and request.end_date is not None:
        if request.end_date < request.start_date:
            collector.error("travel_request", "date_order_invalid", "End date precedes start date.", "end_date")
        else:
            collector.passed("travel_request", "dates_ordered", "Travel dates are valid and ordered.", "start_date/end_date")
        if request.duration is not None:
            expected = (request.end_date - request.start_date).days + 1
            if expected != request.duration:
                collector.error(
                    "travel_request", "duration_date_mismatch", "Duration does not match the inclusive travel-date span.", "duration"
                )
            else:
                collector.passed("travel_request", "duration_matches_dates", "Duration matches the inclusive travel-date span.", "duration")
    elif request.start_date is None or request.end_date is None:
        collector.warning("travel_request", "dates_incomplete", "Complete travel dates were not supplied.", "start_date/end_date")
    if request.budget is not None:
        if request.budget < 0:
            collector.error("travel_request", "budget_negative", "Budget cannot be negative.", "budget")
        else:
            collector.passed("travel_request", "budget_nonnegative", "Supplied budget is non-negative.", "budget")
        if request.currency and len(request.currency) == 3:
            collector.passed("travel_request", "budget_currency_valid", "Budget currency is a valid three-letter code.", "currency")
    return request


def _agent_succeeded(state: Mapping[str, Any], key: str) -> bool:
    """Interpret existing agent completion statuses without equating partial with failure."""
    statuses = state.get("agent_status")
    value = statuses.get(key) if isinstance(statuses, Mapping) else None
    return isinstance(value, str) and value in _SUCCESS_STATUSES


def _validate_research(
    state: Mapping[str, Any], request: TravelRequest | None, collector: _ValidationCollector
) -> DestinationResearch | None:
    """Validate destination research shape, matching destination, duplicate entries and sources."""
    category: ValidationCategory = "destination_research"
    raw = _plain(state.get("destination_research"))
    if raw is None:
        severity: ValidationSeverity = "ERROR" if _agent_succeeded(state, "destination_research") else "WARNING"
        collector.add(category, severity, "research_missing", "Destination research output is unavailable.", "destination_research")
        return None
    try:
        research = DestinationResearch.model_validate(raw)
    except ValidationError as exc:
        collector.error(category, "research_invalid", f"Destination research failed Pydantic validation: {exc}", "destination_research")
        return None

    collector.passed(category, "research_schema_valid", "Destination research satisfies its Pydantic schema.", "destination_research")
    if request and research.destination.strip().casefold() != request.destination.strip().casefold():
        collector.error(category, "research_destination_mismatch", "Research destination does not match the travel request.", "destination_research.destination")
    else:
        collector.passed(category, "research_destination_matches", "Research destination matches the request.", "destination_research.destination")
    if not research.overview:
        collector.warning(category, "research_overview_missing", "Destination overview is unavailable.", "destination_research.overview")
    else:
        collector.passed(category, "research_overview_present", "Destination overview is present.", "destination_research.overview")
    if not research.best_time_to_visit:
        collector.warning(category, "best_time_to_visit_missing", "Best-time-to-visit information is unavailable.", "destination_research.best_time_to_visit")
    if not research.recommended_activities:
        collector.warning(category, "recommended_activities_missing", "Recommended activities are unavailable.", "destination_research.recommended_activities")
    if not research.local_culture:
        collector.warning(category, "local_culture_missing", "Local culture and practical destination context are unavailable.", "destination_research.local_culture")
    if not research.attractions:
        severity = "ERROR" if _agent_succeeded(state, "destination_research") else "WARNING"
        collector.add(category, severity, "attractions_missing", "No researched attractions are available.", "destination_research.attractions")
    else:
        collector.passed(category, "attractions_present", "At least one researched attraction is available.", "destination_research.attractions")

    seen_names: set[str] = set()
    for index, attraction in enumerate(research.attractions):
        name = attraction.name.strip()
        normalized = name.casefold()
        if normalized in seen_names:
            collector.error(category, "duplicate_attraction", f"Duplicate attraction record: {name}.", f"destination_research.attractions[{index}].name")
        else:
            seen_names.add(normalized)
            collector.passed(category, "attraction_name_unique", f"Attraction name {name!r} is valid and unique.", f"destination_research.attractions[{index}].name")
        if (attraction.latitude is None) != (attraction.longitude is None):
            # This is also enforced by Attraction, but keep a defensive rule for mapping inputs.
            collector.error(category, "coordinate_pair_invalid", f"Attraction {name!r} must have both coordinates or neither.", f"destination_research.attractions[{index}]")
        elif attraction.latitude is not None and attraction.longitude is not None:
            if not (-90 <= attraction.latitude <= 90 and -180 <= attraction.longitude <= 180):
                collector.error(category, "coordinates_out_of_range", f"Attraction {name!r} has out-of-range coordinates.", f"destination_research.attractions[{index}]")
            else:
                collector.passed(category, "coordinates_valid", f"Attraction {name!r} coordinates are within geographic ranges.", f"destination_research.attractions[{index}]")
        if attraction.estimated_cost is not None and not (
            attraction.estimated_cost_source_name or attraction.estimated_cost_source_url
        ):
            collector.warning(category, "attraction_cost_source_missing", f"Attraction cost for {name!r} has no specific price source attribution.", f"destination_research.attractions[{index}].estimated_cost")

    if not research.research_sources:
        collector.warning(category, "research_sources_missing", "No source URLs are attached to destination research.", "destination_research.research_sources")
    else:
        invalid = [url for url in research.research_sources if not _valid_url(url)]
        if invalid:
            collector.error(category, "research_source_url_invalid", f"Invalid research source URL(s): {', '.join(invalid)}.", "destination_research.research_sources")
        else:
            collector.passed(category, "research_source_urls_valid", "Destination research source URLs are valid HTTP(S) URLs.", "destination_research.research_sources")
    return research


def _trip_duration(request: TravelRequest | None) -> int | None:
    """Return explicit duration or inclusive date-derived duration."""
    if request is None:
        return None
    if request.duration is not None:
        return request.duration
    if request.start_date is not None and request.end_date is not None and request.end_date >= request.start_date:
        return (request.end_date - request.start_date).days + 1
    return None


def _validate_itinerary(
    state: Mapping[str, Any], request: TravelRequest | None, research: DestinationResearch | None,
    collector: _ValidationCollector,
) -> list[ItineraryDay] | None:
    """Validate itinerary records, day coverage, date consistency, and attraction references."""
    category: ValidationCategory = "itinerary"
    raw = _plain(state.get("itinerary"))
    if raw is None:
        severity: ValidationSeverity = "ERROR" if _agent_succeeded(state, "itinerary_planning") else "WARNING"
        collector.add(category, severity, "itinerary_missing", "Itinerary output is unavailable.", "itinerary")
        return None
    if not isinstance(raw, list):
        collector.error(category, "itinerary_not_list", "Itinerary must be a list of day records.", "itinerary")
        return None

    days: list[ItineraryDay] = []
    for index, raw_day in enumerate(raw):
        try:
            days.append(ItineraryDay.model_validate(_plain(raw_day)))
        except ValidationError as exc:
            collector.error(category, "itinerary_day_invalid", f"Itinerary day at index {index} is invalid: {exc}", f"itinerary[{index}]")
    if len(days) != len(raw):
        return days
    collector.passed(category, "itinerary_schema_valid", "All itinerary days satisfy the ItineraryDay schema.", "itinerary")

    numbers = [day.day_number for day in days]
    if len(numbers) != len(set(numbers)):
        collector.error(category, "itinerary_day_number_duplicate", "Duplicate itinerary day numbers exist.", "itinerary.day_number")
    else:
        collector.passed(category, "itinerary_day_numbers_unique", "Itinerary day numbers are unique.", "itinerary.day_number")
    if numbers and sorted(numbers) != list(range(1, max(numbers) + 1)):
        collector.error(category, "itinerary_day_numbers_not_continuous", "Itinerary day numbering is not continuous from day 1.", "itinerary.day_number")
    elif numbers:
        collector.passed(category, "itinerary_day_numbers_continuous", "Itinerary day numbering is continuous.", "itinerary.day_number")

    dates = [day.date for day in days if day.date is not None]
    if len(dates) != len(set(dates)):
        collector.error(category, "itinerary_date_duplicate", "Multiple itinerary days use the same date.", "itinerary.date")
    elif dates:
        collector.passed(category, "itinerary_dates_unique", "Supplied itinerary dates are unique.", "itinerary.date")
    else:
        collector.warning(category, "itinerary_dates_missing", "No itinerary day dates are supplied.", "itinerary.date")

    duration = _trip_duration(request)
    if duration is not None:
        if len(days) != duration:
            collector.error(category, "itinerary_duration_mismatch", f"Itinerary has {len(days)} days; requested trip duration is {duration}.", "itinerary")
        else:
            collector.passed(category, "itinerary_duration_matches", "Itinerary day count matches the requested trip duration.", "itinerary")
    else:
        collector.warning(category, "itinerary_duration_unknown", "Trip duration cannot be determined for itinerary coverage validation.", "duration")

    if request and request.start_date is not None:
        for day in days:
            expected_date = request.start_date + timedelta(days=day.day_number - 1)
            if day.date is None:
                collector.warning(category, "itinerary_day_date_missing", f"Date for itinerary day {day.day_number} is unavailable.", f"itinerary[{day.day_number - 1}].date")
            elif day.date != expected_date:
                collector.error(category, "itinerary_date_mismatch", f"Day {day.day_number} date does not match the travel request sequence.", f"itinerary[{day.day_number - 1}].date")
            else:
                collector.passed(category, "itinerary_date_matches", f"Day {day.day_number} date matches the travel request.", f"itinerary[{day.day_number - 1}].date")

    research_names = {attraction.name.strip().casefold() for attraction in research.attractions} if research else set()
    if research_names:
        for day_index, day in enumerate(days):
            if not day.attraction_order:
                collector.warning(category, "attraction_order_missing", f"No explicit attraction order is recorded for day {day.day_number}; not all researched attractions are required to be visited.", f"itinerary[{day_index}].attraction_order")
            seen_refs: set[str] = set()
            for name in day.attraction_order:
                normalized = name.strip().casefold()
                if normalized in seen_refs:
                    collector.warning(category, "duplicate_attraction_reference", f"Attraction {name!r} is repeated in day {day.day_number}; repeated visits cannot be distinguished from duplicate references by the current schema.", f"itinerary[{day_index}].attraction_order")
                seen_refs.add(normalized)
                if normalized not in research_names:
                    collector.error(category, "attraction_reference_invalid", f"Itinerary references unresearched attraction {name!r}.", f"itinerary[{day_index}].attraction_order")
                else:
                    collector.passed(category, "attraction_reference_valid", f"Attraction reference {name!r} exists in destination research.", f"itinerary[{day_index}].attraction_order")
    else:
        collector.warning(category, "attraction_reference_check_unavailable", "Attraction references could not be cross-checked without destination research.", "itinerary.attraction_order")

    for day_index, day in enumerate(days):
        if not day.activities:
            collector.warning(category, "itinerary_activities_empty", f"Day {day.day_number} has no activity descriptions.", f"itinerary[{day_index}].activities")

    for index, day in enumerate(days):
        distance = day.estimated_straight_line_distance_km
        if distance is not None:
            if not math.isfinite(distance) or distance < 0:
                collector.error(category, "route_distance_invalid", f"Day {day.day_number} route distance must be finite and non-negative.", f"itinerary[{index}].estimated_straight_line_distance_km")
            else:
                collector.passed(category, "route_distance_valid", f"Day {day.day_number} straight-line distance is non-negative.", f"itinerary[{index}].estimated_straight_line_distance_km")
    return days


def _validate_budget(
    state: Mapping[str, Any], request: TravelRequest | None, collector: _ValidationCollector
) -> tuple[BudgetBreakdown | None, bool | None]:
    """Validate budget structures and independently recalculate included line items."""
    category: ValidationCategory = "budget"
    raw = _plain(state.get("budget_breakdown"))
    if raw is None:
        severity: ValidationSeverity = "ERROR" if _agent_succeeded(state, "budget_planning") else "WARNING"
        collector.add(category, severity, "budget_missing", "Budget breakdown is unavailable.", "budget_breakdown")
        return None, None

    # Perform minimal raw arithmetic before Pydantic validation. This allows
    # the report to identify an independent subtotal/currency inconsistency
    # even when the schema's own consistency validators reject the object.
    if isinstance(raw, Mapping):
        raw_lines = raw.get("line_items")
        if isinstance(raw_lines, list):
            raw_subtotal = _ZERO
            raw_currencies: set[str] = set()
            raw_lines_valid = True
            for index, raw_item in enumerate(raw_lines):
                if not isinstance(raw_item, Mapping):
                    raw_lines_valid = False
                    continue
                if not raw_item.get("included_in_total"):
                    continue
                try:
                    unit_price = _decimal(raw_item.get("unit_price"), f"line_items[{index}].unit_price")
                    quantity = _decimal(raw_item.get("quantity"), f"line_items[{index}].quantity")
                    if unit_price < _ZERO or quantity < _ZERO:
                        raise ValueError("unit price and quantity must be non-negative")
                    raw_subtotal += unit_price * quantity
                    currency_value = raw_item.get("currency")
                    if isinstance(currency_value, str):
                        raw_currencies.add(currency_value.strip().upper())
                except (ValueError, InvalidOperation, TypeError) as exc:
                    raw_lines_valid = False
                    collector.error(category, "budget_line_numeric_invalid", str(exc), f"budget_breakdown.line_items[{index}]")
            if raw_lines_valid:
                try:
                    supplied_subtotal = _money(_decimal(raw.get("known_subtotal", "0"), "Known subtotal"))
                    if abs(supplied_subtotal - _money(raw_subtotal)) > _CENT:
                        collector.error(category, "budget_subtotal_mismatch", "Known subtotal does not match the independently summed included expense lines.", "budget_breakdown.known_subtotal")
                except (ValueError, InvalidOperation, TypeError) as exc:
                    collector.error(category, "budget_subtotal_invalid", str(exc), "budget_breakdown.known_subtotal")
            if len(raw_currencies) > 1:
                collector.error(category, "budget_mixed_currencies", "Included budget line items use multiple currencies.", "budget_breakdown.line_items.currency")
    try:
        budget = BudgetBreakdown.model_validate(raw)
    except ValidationError as exc:
        collector.error(category, "budget_invalid", f"Budget breakdown failed Pydantic validation: {exc}", "budget_breakdown")
        return None, None
    collector.passed(category, "budget_schema_valid", "Budget breakdown satisfies its Pydantic schema.", "budget_breakdown")

    request_currency = request.currency.upper() if request else None
    if request_currency and budget.currency != request_currency:
        collector.error(category, "budget_currency_request_mismatch", "Budget currency does not match request currency; no conversion is performed.", "budget_breakdown.currency")

    line_items: list[BudgetLineItem] = []
    for index, raw_item in enumerate(budget.line_items):
        try:
            item = BudgetLineItem.model_validate(_plain(raw_item))
            line_items.append(item)
        except ValidationError as exc:
            collector.error(category, "budget_line_item_invalid", f"Budget line item {index} is invalid: {exc}", f"budget_breakdown.line_items[{index}]")
    if len(line_items) != len(budget.line_items):
        return budget, None

    included_by_category: dict[str, Decimal] = {}
    currencies: set[str] = set()
    for index, item in enumerate(line_items):
        field = f"budget_breakdown.line_items[{index}]"
        try:
            amount = _decimal(item.unit_price, f"{item.name} unit price")
            if amount < _ZERO:
                raise ValueError("unit price cannot be negative")
            if item.quantity is None or not _decimal(item.quantity, f"{item.name} quantity").is_finite():
                raise ValueError("included line item requires a finite quantity")
            quantity = _decimal(item.quantity, f"{item.name} quantity")
            if quantity < _ZERO:
                raise ValueError("quantity cannot be negative")
            expected_line_amount = _money(amount * quantity)
            if item.included_in_total:
                currencies.add(item.currency.upper())
                if item.calculated_amount is None:
                    collector.error(category, "budget_line_amount_missing", f"Included line {item.name!r} has no calculated amount.", f"{field}.calculated_amount")
                    continue
                not_trustworthy = (
                    item.price_type == "published"
                    and (item.verification_status != "verified" or not (item.source_name or item.source_url))
                ) or (
                    item.price_type == "estimate" and item.verification_status != "approved"
                ) or item.verification_status in {"unverified", "manual_review"}
                if not_trustworthy:
                    collector.error(category, "unverified_line_included", f"Unverified, unsourced, or unapproved line {item.name!r} is included in totals.", f"{field}.included_in_total")
                actual_line_amount = _money(_decimal(item.calculated_amount, f"{item.name} calculated amount"))
                if abs(actual_line_amount - expected_line_amount) > _CENT:
                    collector.error(category, "budget_line_calculation_mismatch", f"Line amount for {item.name!r} differs from unit price × quantity.", f"{field}.calculated_amount")
                else:
                    collector.passed(category, "budget_line_calculation_valid", f"Line amount for {item.name!r} matches unit price × quantity.", f"{field}.calculated_amount")
                if not not_trustworthy:
                    included_by_category[item.category] = included_by_category.get(item.category, _ZERO) + amount * quantity
            else:
                if item.calculated_amount is not None:
                    collector.error(category, "excluded_line_has_amount", f"Excluded line {item.name!r} must not present a calculated amount as included.", f"{field}.calculated_amount")
                if not item.exclusion_reason:
                    collector.warning(category, "excluded_line_reason_missing", f"Excluded line {item.name!r} has no explanation.", f"{field}.exclusion_reason")
        except (ValueError, InvalidOperation, TypeError) as exc:
            collector.error(category, "budget_line_numeric_invalid", f"Invalid numeric line item {item.name!r}: {exc}", field)

    if len(currencies) > 1:
        collector.error(category, "budget_mixed_currencies", "Included budget line items use multiple currencies.", "budget_breakdown.line_items.currency")
    elif currencies and currencies != {budget.currency.upper()}:
        collector.error(category, "budget_line_currency_mismatch", "Line-item currency does not match budget currency.", "budget_breakdown.line_items.currency")
    else:
        collector.passed(category, "budget_currency_consistent", "Budget and included line-item currencies are consistent.", "budget_breakdown.currency")

    category_values = {
        "accommodation": budget.accommodation_cost,
        "food": budget.food_cost,
        "local_transportation": budget.local_transportation_cost,
        "intercity_transportation": budget.intercity_transportation_cost,
        "activity": budget.activity_cost,
        "other_expenses": budget.miscellaneous_cost,
    }
    for category_name, value in category_values.items():
        expected = _money(included_by_category.get(category_name, _ZERO))
        if value is None:
            if expected != _ZERO or category_name in included_by_category:
                collector.error(category, "budget_category_total_missing", f"{category_name} has included lines but no category total.", f"budget_breakdown.{category_name}")
            continue
        actual = _money(_decimal(value, f"{category_name} total"))
        if abs(actual - expected) > _CENT:
            collector.error(category, "budget_category_total_mismatch", f"{category_name} total does not match included line items.", f"budget_breakdown.{category_name}")
        else:
            collector.passed(category, "budget_category_total_valid", f"{category_name} total matches included line items.", f"budget_breakdown.{category_name}")

    local = _money(included_by_category.get("local_transportation", _ZERO))
    intercity = _money(included_by_category.get("intercity_transportation", _ZERO))
    transport_expected = local + intercity
    has_transport_lines = "local_transportation" in included_by_category or "intercity_transportation" in included_by_category
    if has_transport_lines and budget.transportation_cost is None:
        collector.error(category, "transportation_total_missing", "Transport line items exist but combined transportation total is missing.", "budget_breakdown.transportation_cost")
    elif budget.transportation_cost is not None and abs(_money(budget.transportation_cost) - transport_expected) > _CENT:
        collector.error(category, "transportation_total_mismatch", "Combined transportation total does not equal local plus intercity transportation.", "budget_breakdown.transportation_cost")
    elif budget.transportation_cost is not None:
        collector.passed(category, "transportation_total_valid", "Combined transportation total matches local plus intercity costs.", "budget_breakdown.transportation_cost")

    subtotal_raw = sum(included_by_category.values(), _ZERO)
    subtotal = _money(subtotal_raw)
    if abs(_money(budget.known_subtotal) - subtotal) > _CENT:
        collector.error(category, "budget_subtotal_mismatch", "Known subtotal does not match the independently summed included expense lines.", "budget_breakdown.known_subtotal")
    else:
        collector.passed(category, "budget_subtotal_valid", "Known subtotal matches included expense lines.", "budget_breakdown.known_subtotal")

    percent = _decimal(budget.emergency_buffer_percent, "Emergency buffer percentage")
    configured_buffer = state.get("emergency_buffer_percent")
    if configured_buffer is not None:
        try:
            configured_decimal = _decimal(configured_buffer, "Configured emergency buffer percentage")
            if configured_decimal != percent:
                collector.error(category, "configured_buffer_mismatch", "Budget buffer percentage does not match the explicitly configured TravelState percentage.", "budget_breakdown.emergency_buffer_percent")
        except ValueError as exc:
            collector.error(category, "configured_buffer_invalid", str(exc), "emergency_buffer_percent")
    if percent < 0 or percent > 100:
        collector.error(category, "buffer_percent_invalid", "Emergency buffer percentage must be between 0 and 100.", "budget_breakdown.emergency_buffer_percent")
        expected_buffer = _ZERO
    else:
        expected_buffer = _money(subtotal_raw * percent / Decimal("100"))
    if abs(_money(budget.emergency_buffer) - expected_buffer) > _CENT:
        collector.error(category, "emergency_buffer_mismatch", "Emergency buffer does not match the configured percentage and known subtotal.", "budget_breakdown.emergency_buffer")
    else:
        collector.passed(category, "emergency_buffer_valid", "Emergency buffer matches the documented subtotal × percentage calculation.", "budget_breakdown.emergency_buffer")

    expected_total = _money(subtotal_raw + expected_buffer)
    if budget.total_estimated_cost is None:
        collector.warning(category, "budget_total_missing", "Final estimated total is unavailable.", "budget_breakdown.total_estimated_cost")
    elif abs(_money(budget.total_estimated_cost) - expected_total) > _CENT:
        collector.error(category, "budget_total_mismatch", "Final total does not match known subtotal plus emergency buffer.", "budget_breakdown.total_estimated_cost")
    else:
        collector.passed(category, "budget_total_valid", "Final total matches known subtotal plus emergency buffer.", "budget_breakdown.total_estimated_cost")

    travelers = request.travelers if request else None
    if travelers and budget.total_estimated_cost is not None:
        expected_per_person = _money(expected_total / Decimal(travelers))
        if budget.estimated_cost_per_person is None or abs(_money(budget.estimated_cost_per_person) - expected_per_person) > _CENT:
            collector.error(category, "cost_per_person_mismatch", "Per-person cost does not match final total divided by traveler count.", "budget_breakdown.estimated_cost_per_person")
        else:
            collector.passed(category, "cost_per_person_valid", "Per-person cost matches final total divided by traveler count.", "budget_breakdown.estimated_cost_per_person")

    budget_within: bool | None = None
    user_budget = request.budget if request and request.budget is not None else budget.user_budget
    if request and request.budget is not None and budget.user_budget is not None:
        if _money(_decimal(request.budget, "Travel request budget")) != _money(_decimal(budget.user_budget, "Budget breakdown user budget")):
            collector.error(category, "user_budget_mismatch", "Budget breakdown user_budget differs from the travel request budget.", "budget_breakdown.user_budget")
    missing_categories = set(budget.missing_categories)
    included_categories = set(included_by_category)
    expected_missing = {"accommodation", "food", "local_transportation", "intercity_transportation", "activity", "other_expenses"} - included_categories
    if not expected_missing.issubset(missing_categories):
        collector.error(category, "budget_missing_categories_mismatch", "Missing expense categories are not fully represented in the budget completeness list.", "budget_breakdown.missing_categories")
    if budget.estimate_status in {"complete", "assumption_based"} and (missing_categories or expected_missing):
        collector.error(category, "budget_partial_marked_complete", "Budget is marked complete despite missing price categories.", "budget_breakdown.estimate_status")
    elif budget.estimate_status in {"partial", "partial_with_assumptions"} and (missing_categories or expected_missing):
        collector.warning(category, "budget_estimate_partial", "Budget total covers only known prices; comparison is provisional.", "budget_breakdown.estimate_status")
    else:
        collector.passed(category, "budget_completeness_consistent", "Budget completeness status matches listed categories.", "budget_breakdown.estimate_status")

    if user_budget is not None and budget.total_estimated_cost is not None:
        budget_decimal = _decimal(user_budget, "User budget")
        if budget_decimal <= _ZERO:
            collector.error(category, "budget_limit_invalid", "User budget must be positive for comparison.", "budget")
        else:
            expected_remaining = _money(budget_decimal - expected_total)
            expected_utilization = (expected_total / budget_decimal * Decimal("100")).quantize(_CENT, rounding=ROUND_HALF_UP)
            if budget.remaining_budget is None or abs(_money(budget.remaining_budget) - expected_remaining) > _CENT:
                collector.error(category, "remaining_budget_mismatch", "Remaining budget does not match user budget minus estimated total.", "budget_breakdown.remaining_budget")
            else:
                collector.passed(category, "remaining_budget_valid", "Remaining budget matches user budget minus estimated total.", "budget_breakdown.remaining_budget")
            if budget.budget_utilization_percent is None or abs(_decimal(budget.budget_utilization_percent, "Budget utilization") - expected_utilization) > _CENT:
                collector.error(category, "budget_utilization_mismatch", "Budget utilization does not match estimated total divided by user budget.", "budget_breakdown.budget_utilization_percent")
            else:
                collector.passed(category, "budget_utilization_valid", "Budget utilization matches the independent calculation.", "budget_breakdown.budget_utilization_percent")
            complete = budget.estimate_status in {"complete", "assumption_based"} and not missing_categories and not expected_missing
            budget_within = expected_total <= budget_decimal if complete else None
            expected_comparison = (
                "provisional" if not complete else ("within_limit" if expected_total <= budget_decimal else "over_budget")
            )
            if budget.budget_comparison_status != expected_comparison:
                collector.error(category, "budget_comparison_status_mismatch", "Budget comparison status is inconsistent with amount and estimate completeness.", "budget_breakdown.budget_comparison_status")
    elif user_budget is None:
        collector.warning(category, "user_budget_missing", "No user budget was supplied; budget-limit comparison is unavailable.", "budget")

    return budget, budget_within


def _validate_advice_items(
    items: list[AdviceItem], collector: _ValidationCollector, field: str, *, category: ValidationCategory = "travel_information"
) -> None:
    """Check advice descriptions, URLs, and verified-source basis metadata."""
    for index, item in enumerate(items):
        location = f"{field}[{index}]"
        if not item.text.strip():
            collector.error(category, "advice_text_missing", "Advice description is empty.", f"{location}.text")
        else:
            collector.passed(category, "advice_text_valid", "Advice description is present.", f"{location}.text")
        for source_index, source in enumerate(item.sources):
            if source.url and not _valid_url(source.url):
                collector.error(category, "advice_source_invalid", f"Advice contains invalid source URL {source.url!r}.", f"{location}.sources[{source_index}].url")
        if item.basis == "verified_source" and not item.sources:
            collector.error(category, "verified_advice_source_missing", "Advice marked verified_source has no source reference.", f"{location}.sources")
        if item.basis == "forecast" and not any(
            source.information_type == "forecast" for source in item.sources
        ):
            collector.error(category, "forecast_advice_source_missing", "Forecast-based advice does not cite a forecast source.", f"{location}.sources")


def _validate_travel_information(
    state: Mapping[str, Any], request: TravelRequest | None, collector: _ValidationCollector
) -> TravelInformation | None:
    """Validate structured travel assistant information without fact-checking its claims."""
    category: ValidationCategory = "travel_information"
    raw = _plain(state.get("travel_information"))
    if raw is None:
        severity: ValidationSeverity = "ERROR" if _agent_succeeded(state, "travel_assistant") else "WARNING"
        collector.add(category, severity, "travel_information_missing", "Travel information output is unavailable.", "travel_information")
        return None
    try:
        info = TravelInformation.model_validate(raw)
    except ValidationError as exc:
        collector.error(category, "travel_information_invalid", f"Travel information failed Pydantic validation: {exc}", "travel_information")
        return None
    collector.passed(category, "travel_information_schema_valid", "Travel information satisfies its Pydantic schema.", "travel_information")

    weather = info.weather
    if weather is None:
        collector.warning(category, "weather_missing", "Structured weather forecast is unavailable.", "travel_information.weather")
    else:
        if weather.status == "unavailable" and not weather.unavailable_reason:
            collector.warning(category, "weather_unavailable_reason_missing", "Weather is marked unavailable without a reason.", "travel_information.weather.unavailable_reason")
        for index, observation in enumerate(weather.daily_forecasts):
            field = f"travel_information.weather.daily_forecasts[{index}]"
            if observation.temperature_min_c is not None and not math.isfinite(observation.temperature_min_c):
                collector.error(category, "weather_temperature_invalid", "Minimum temperature must be finite.", f"{field}.temperature_min_c")
            if observation.temperature_max_c is not None and not math.isfinite(observation.temperature_max_c):
                collector.error(category, "weather_temperature_invalid", "Maximum temperature must be finite.", f"{field}.temperature_max_c")
            if observation.temperature_min_c is not None and observation.temperature_max_c is not None and observation.temperature_min_c > observation.temperature_max_c:
                collector.error(category, "weather_temperature_range_invalid", "Minimum forecast temperature exceeds maximum.", field)
            if observation.precipitation_mm is not None and (not math.isfinite(observation.precipitation_mm) or observation.precipitation_mm < 0):
                collector.error(category, "weather_precipitation_invalid", "Precipitation must be finite and non-negative.", f"{field}.precipitation_mm")
            if observation.source.information_type != "forecast":
                collector.error(category, "weather_source_type_invalid", "Weather observation source must be marked as a forecast.", f"{field}.source")
            if request and request.start_date and request.end_date and not (request.start_date <= observation.date <= request.end_date):
                collector.error(category, "weather_date_outside_trip", "Forecast date is outside the requested trip date range.", f"{field}.date")
            else:
                collector.passed(category, "weather_observation_valid", f"Weather observation for {observation.date} has a valid structure and trip date.", field)
        if request and request.start_date and request.end_date:
            covered_dates = {item.date for item in weather.daily_forecasts}
            covered_dates.update(weather.unavailable_dates)
            expected_dates = {
                request.start_date + timedelta(days=offset)
                for offset in range((request.end_date - request.start_date).days + 1)
            }
            missing_coverage = expected_dates - covered_dates
            if missing_coverage:
                collector.warning(
                    category,
                    "weather_coverage_incomplete",
                    f"Weather coverage does not identify {len(missing_coverage)} requested date(s) as forecasted or unavailable.",
                    "travel_information.weather",
                )
            invalid_unavailable = [day for day in weather.unavailable_dates if not request.start_date <= day <= request.end_date]
            if invalid_unavailable:
                collector.error(category, "weather_unavailable_date_outside_trip", "Unavailable forecast dates include dates outside the trip.", "travel_information.weather.unavailable_dates")
            elif weather.unavailable_dates:
                collector.warning(category, "weather_dates_unavailable", f"Forecast unavailable for {len(weather.unavailable_dates)} travel date(s).", "travel_information.weather.unavailable_dates")
            else:
                collector.passed(category, "weather_trip_dates_covered", "No trip dates are explicitly marked forecast-unavailable.", "travel_information.weather.unavailable_dates")
        if weather.status == "unavailable":
            collector.warning(category, "weather_not_available", "No actual weather forecast is available; climate averages are not substituted.", "travel_information.weather.status")

    transport = info.transportation
    if transport is None:
        collector.warning(category, "transportation_missing", "Structured transport and logistics information is unavailable.", "travel_information.transportation")
    else:
        advice = transport.local_options + transport.intercity_options + transport.logistics_tips
        _validate_advice_items(advice, collector, "travel_information.transportation")
        for index, item in enumerate(advice):
            if item.basis == "verified_source" and _EXACT_TRANSPORT_CLAIM.search(item.text):
                collector.warning(
                    category,
                    "exact_transport_claim_not_independently_verifiable",
                    "A specific fare, schedule, duration, or distance appears in verified transport text; URL metadata alone cannot independently verify the claim.",
                    f"travel_information.transportation[{index}]",
                )
        if transport.status == "verified" and any(not item.sources for item in advice):
            collector.warning(category, "transport_verified_source_missing", "Some transport information marked verified has no source.", "travel_information.transportation")
        if not advice:
            collector.warning(category, "transport_options_missing", "No transport options or logistics tips were provided.", "travel_information.transportation")

    packing = info.packing
    if packing is None:
        collector.warning(category, "packing_missing", "Packing recommendations are unavailable.", "travel_information.packing")
    else:
        packing_items = (
            packing.clothing + packing.footwear + packing.personal_essentials
            + packing.travel_documents + packing.electronics_accessories
            + packing.weather_specific + packing.activity_specific
        )
        _validate_advice_items(packing_items, collector, "travel_information.packing")
        has_forecast_items = any(item.basis == "forecast" for item in packing.weather_specific)
        forecast_available = bool(weather and weather.daily_forecasts)
        if has_forecast_items and not forecast_available:
            collector.error(category, "packing_forecast_without_weather", "Forecast-based packing advice exists without any actual forecast observations.", "travel_information.packing.weather_specific")
        if packing.forecast_based and not has_forecast_items:
            collector.error(category, "packing_forecast_flag_mismatch", "Packing is marked forecast-based but contains no forecast-based weather items.", "travel_information.packing.forecast_based")
        if not packing_items:
            collector.warning(category, "packing_items_missing", "No categorized packing recommendations were provided.", "travel_information.packing")

    _validate_advice_items(info.practical_tips, collector, "travel_information.practical_tips")
    if not info.practical_tips and not info.safety_tips:
        collector.warning(category, "practical_tips_missing", "Practical or safety tips are unavailable.", "travel_information.practical_tips")
    for index, source in enumerate(info.sources):
        if source.url and not _valid_url(source.url):
            collector.error(category, "travel_source_url_invalid", f"Invalid travel information source URL {source.url!r}.", f"travel_information.sources[{index}].url")
    return info


def validate_travel_state(state: TravelState) -> ValidationResult:
    """Run independent deterministic rules and return a structured report."""
    collector = _ValidationCollector()
    partial = False
    try:
        request = _validate_request(state, collector)
        partial = request is None
        research = _validate_research(state, request, collector)
        itinerary = _validate_itinerary(state, request, research, collector)
        budget, budget_within_limit = _validate_budget(state, request, collector)
        travel_information = _validate_travel_information(state, request, collector)
        del budget, travel_information  # reports are represented in collected issues only
    except Exception as exc:  # noqa: BLE001 - validation must never destroy/stop the plan.
        logger.exception("Unexpected validation rule failure")
        collector.error("travel_request", "validation_internal_error", f"Validation could not complete one or more checks ({type(exc).__name__}): {exc}")
        request = None
        itinerary = None
        budget_within_limit = None

    statuses = state.get("agent_status")
    for output_key, agent_key, display in (
        ("destination_research", "destination_research", "destination research"),
        ("itinerary", "itinerary_planning", "itinerary"),
        ("budget_breakdown", "budget_planning", "budget breakdown"),
        ("travel_information", "travel_assistant", "travel information"),
    ):
        if state.get(output_key) is None:
            partial = True
            agent_status = statuses.get(agent_key) if isinstance(statuses, Mapping) else None
            collector.warning(
                "destination_research" if output_key == "destination_research" else "itinerary" if output_key == "itinerary" else "budget" if output_key == "budget_breakdown" else "travel_information",
                f"{agent_key}_output_unavailable",
                f"Validation is partial because {display} is unavailable (agent status: {agent_status or 'not recorded'}).",
                output_key,
            )

    errors = sum(issue.severity == "ERROR" for issue in collector.issues)
    warnings = sum(issue.severity == "WARNING" for issue in collector.issues)
    passed = sum(issue.severity == "PASS" for issue in collector.issues)
    if errors:
        status = "FAILED"
    elif partial:
        status = "PARTIAL"
    elif warnings:
        status = "PASSED_WITH_WARNINGS"
    else:
        status = "PASSED"
    return ValidationResult(
        status=status,
        is_valid=errors == 0,
        checks_performed=len(collector.issues),
        checks_passed=passed,
        error_count=errors,
        warning_count=warnings,
        issues=collector.issues,
        suggestions=[],
        budget_within_limit=budget_within_limit,
        itinerary_feasible=(
            not any(issue.category == "itinerary" and issue.severity == "ERROR" for issue in collector.issues)
            if itinerary is not None
            else None
        ),
    )


def validation_agent(state: TravelState) -> dict[str, Any]:
    """Validate existing plan outputs and return only validation/status/errors.

    This node is pure Python/Pydantic. It never uses an LLM and never repairs or
    rewrites research, itinerary, budget, or travel information values.
    """
    try:
        result = validate_travel_state(state)
    except Exception as exc:  # noqa: BLE001 - last safety boundary for graph execution.
        logger.exception("Validation agent failed unexpectedly")
        return {
            "agent_status": {_AGENT_KEY: "failed"},
            "errors": [f"Validation agent failed ({type(exc).__name__}): {exc}"],
        }

    status_map = {
        "PASSED": "completed",
        "PASSED_WITH_WARNINGS": "completed_with_warnings",
        "FAILED": "failed",
        "PARTIAL": "completed_partial",
    }
    update: dict[str, Any] = {
        "validation_result": model_to_state_update(result),
        "agent_status": {_AGENT_KEY: status_map[result.status]},
    }
    errors = [issue.message for issue in result.issues if issue.severity == "ERROR"]
    if errors:
        update["errors"] = errors
    return update
