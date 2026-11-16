"""Reusable Pydantic v2 schemas for travel requests and agent outputs."""

from __future__ import annotations

from datetime import date as DateType
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SchemaModel(BaseModel):
    """Base configuration shared by the public travel data models."""

    model_config = ConfigDict(extra="forbid")


BudgetCategory = Literal[
    "accommodation",
    "food",
    "local_transportation",
    "intercity_transportation",
    "activity",
    "other_expenses",
]
BudgetUnit = Literal[
    "per_room_per_night",
    "per_person_per_day",
    "per_day",
    "per_trip",
    "per_vehicle",
    "per_vehicle_trip",
    "per_person_ticket",
    "per_item",
    "flat",
]
BudgetPriceType = Literal["user_provided", "published", "estimate"]
BudgetVerificationStatus = Literal[
    "user_supplied", "verified", "approved", "unverified", "manual_review"
]


class TravelRequest(SchemaModel):
    """Validated user requirements used to initialize a travel-planning run."""

    origin: str | None = Field(default=None, description="Starting location.")
    destination: str = Field(
        ..., min_length=1, description="Target destination; must not be blank."
    )
    start_date: DateType | None = Field(default=None, description="First travel date.")
    end_date: DateType | None = Field(default=None, description="Last travel date.")
    duration: int | None = Field(
        default=None, gt=0, description="Trip duration in calendar travel days."
    )
    budget: float | None = Field(
        default=None, gt=0, description="Maximum total budget; no currency conversion is applied."
    )
    currency: str = Field(
        default="INR", min_length=3, max_length=3, description="Three-letter currency code."
    )
    travelers: int = Field(default=1, ge=1, description="Number of travelers.")
    preferences: list[str] = Field(
        default_factory=list, description="Interests and travel preferences."
    )
    travel_style: Literal["budget", "moderate", "luxury"] | None = Field(
        default=None, description="Preferred overall travel style."
    )
    additional_requirements: str | None = Field(
        default=None, description="Other constraints or requirements supplied by the user."
    )

    @field_validator("destination")
    @classmethod
    def destination_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only destinations and normalize surrounding spaces."""
        value = value.strip()
        if not value:
            raise ValueError("Destination must not be empty.")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize a three-character currency code to uppercase."""
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("Currency must be a three-letter code.")
        return value

    @field_validator("origin")
    @classmethod
    def normalize_origin(cls, value: str | None) -> str | None:
        """Normalize a supplied origin while allowing it to be omitted."""
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_trip_dates_and_duration(self) -> TravelRequest:
        """Ensure dates are ordered and agree with the inclusive day count."""
        if self.start_date is not None and self.end_date is not None:
            if self.end_date < self.start_date:
                raise ValueError("End date must not precede start date.")
            if self.duration is not None:
                actual_duration = (self.end_date - self.start_date).days + 1
                if self.duration != actual_duration:
                    raise ValueError(
                        "Duration must match the inclusive number of days between the dates."
                    )
        return self


class Attraction(SchemaModel):
    """A researched attraction; unknown cost or duration remains unspecified."""

    name: str = Field(..., min_length=1, description="Attraction name.")
    description: str | None = Field(default=None, description="Short factual description.")
    location: str | None = Field(default=None, description="Address, area, or coordinates.")
    category: str | None = Field(default=None, description="Attraction category.")
    estimated_cost: float | None = Field(
        default=None, ge=0, description="Known estimated cost per traveler, if available."
    )
    estimated_cost_currency: str | None = Field(
        default=None, description="Currency for the attraction cost, if explicitly known."
    )
    estimated_cost_pricing_unit: Literal["per_person_ticket"] | None = Field(
        default=None, description="Explicit unit for the attraction cost."
    )
    estimated_cost_source_name: str | None = Field(
        default=None, description="Specific source for the attraction cost."
    )
    estimated_cost_source_url: str | None = Field(
        default=None, description="URL supporting the attraction cost."
    )
    estimated_cost_price_type: BudgetPriceType | None = Field(
        default=None, description="Whether the attraction cost is published, user-provided, or an estimate."
    )
    estimated_cost_verification_status: BudgetVerificationStatus | None = Field(
        default=None, description="Verification state of the attraction cost."
    )
    recommended_duration: str | None = Field(
        default=None, description="Suggested visit length, if known."
    )
    latitude: float | None = Field(
        default=None, ge=-90, le=90, description="Verified latitude in decimal degrees."
    )
    longitude: float | None = Field(
        default=None, ge=-180, le=180, description="Verified longitude in decimal degrees."
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only names."""
        value = value.strip()
        if not value:
            raise ValueError("Attraction name must not be empty.")
        return value

    @model_validator(mode="after")
    def coordinates_must_be_paired(self) -> Attraction:
        """Require both coordinates together; neither is fabricated by default."""
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Latitude and longitude must both be provided or both be absent.")
        return self

    @field_validator("estimated_cost_currency")
    @classmethod
    def normalize_estimated_cost_currency(cls, value: str | None) -> str | None:
        """Normalize an explicitly supplied attraction-price currency."""
        if value is None:
            return None
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("Estimated cost currency must be a three-letter code.")
        return value


class DestinationResearch(SchemaModel):
    """Structured destination research returned by a future research agent."""

    destination: str = Field(..., min_length=1, description="Destination researched.")
    overview: str | None = Field(default=None, description="Research-backed overview.")
    best_time_to_visit: str | None = Field(default=None, description="Research-backed season guidance.")
    attractions: list[Attraction] = Field(
        default_factory=list, description="Known attractions for the destination."
    )
    recommended_activities: list[str] = Field(
        default_factory=list, description="Recommended activities supported by research."
    )
    local_culture: str | None = Field(default=None, description="Relevant local customs and culture.")
    research_sources: list[str] = Field(
        default_factory=list, description="URLs or citation labels supporting the research."
    )


class ItineraryDay(SchemaModel):
    """One day of a proposed itinerary."""

    day_number: int = Field(..., ge=1, description="One-based day number in the trip.")
    date: DateType | None = Field(default=None, description="Calendar date, when known.")
    title: str = Field(..., min_length=1, description="Short title for the day.")
    activities: list[str] = Field(default_factory=list, description="Planned activities.")
    estimated_daily_cost: float | None = Field(
        default=None, ge=0, description="Estimated cost for this day, if known."
    )
    travel_notes: list[str] = Field(
        default_factory=list, description="Logistics and other day-specific notes."
    )
    attraction_order: list[str] = Field(
        default_factory=list,
        description="Recommended attraction visiting order; not a road-route guarantee.",
    )
    geographic_optimization_status: Literal["optimized", "limited", "unavailable"] | None = Field(
        default=None,
        description="Availability of straight-line geographic ordering for this day.",
    )
    estimated_straight_line_distance_km: float | None = Field(
        default=None,
        ge=0,
        description="Approximate straight-line route distance in kilometers, not road distance.",
    )
    geographic_notes: list[str] = Field(
        default_factory=list,
        description="Limitations, such as attractions without verified coordinates.",
    )


class BudgetPriceInput(SchemaModel):
    """An explicitly supplied price and its unit, provenance, and needed counts."""

    category: BudgetCategory = Field(..., description="Expense category for this price.")
    name: str = Field(..., min_length=1, description="Identifiable expense name.")
    unit_price: Decimal = Field(..., ge=Decimal("0"), description="Price for one declared unit.")
    currency: str = Field(..., min_length=3, max_length=3, description="Three-letter currency code.")
    pricing_unit: BudgetUnit = Field(..., description="Explicit unit associated with unit_price.")
    rooms: Decimal | None = Field(default=None, ge=Decimal("0"), description="Explicit room count.")
    nights: Decimal | None = Field(default=None, ge=Decimal("0"), description="Explicit night count.")
    trips: Decimal | None = Field(default=None, ge=Decimal("0"), description="Explicit trip count.")
    vehicles: Decimal | None = Field(default=None, ge=Decimal("0"), description="Explicit vehicle count.")
    applicable_travelers: Decimal | None = Field(
        default=None, ge=Decimal("0"), description="Applicable adult/child/group count."
    )
    visits: Decimal | None = Field(default=None, ge=Decimal("0"), description="Explicit ticket visits.")
    quantity: Decimal | None = Field(
        default=None, ge=Decimal("0"), description="Explicit item count for per-item pricing."
    )
    source_name: str | None = Field(default=None, description="Price source or user attribution.")
    source_url: str | None = Field(default=None, description="URL supporting a published price.")
    price_type: BudgetPriceType = Field(..., description="Origin/classification of the price.")
    verification_status: BudgetVerificationStatus = Field(
        default="user_supplied", description="Verification or approval state for the price."
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize and validate a three-letter currency code."""
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("Currency must be a three-letter code.")
        return value

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only expense names."""
        value = value.strip()
        if not value:
            raise ValueError("Expense name must not be empty.")
        return value

class BudgetLineItem(SchemaModel):
    """Auditable calculation record for one supplied expense input."""

    category: BudgetCategory = Field(..., description="Expense category.")
    name: str = Field(..., description="Expense name.")
    unit_price: Decimal = Field(..., ge=Decimal("0"), description="Price per declared unit.")
    currency: str = Field(..., description="Price currency.")
    pricing_unit: BudgetUnit = Field(..., description="Unit for the price.")
    quantity: Decimal | None = Field(default=None, ge=Decimal("0"), description="Calculated quantity.")
    calculated_amount: Decimal | None = Field(
        default=None, ge=Decimal("0"), description="Rounded extended cost when included."
    )
    source_name: str | None = Field(default=None, description="Price source or user attribution.")
    source_url: str | None = Field(default=None, description="Source URL, if supplied.")
    price_type: BudgetPriceType = Field(..., description="Origin/classification of the price.")
    verification_status: BudgetVerificationStatus = Field(..., description="Price verification state.")
    included_in_total: bool = Field(..., description="Whether this price was used in calculations.")
    exclusion_reason: str | None = Field(default=None, description="Why a price was excluded.")


class BudgetBreakdown(SchemaModel):
    """A backward-compatible, auditable budget with completeness metadata."""

    accommodation_cost: Decimal | None = Field(default=None, ge=Decimal("0"), description="Known accommodation cost.")
    food_cost: Decimal | None = Field(default=None, ge=Decimal("0"), description="Known food cost.")
    transportation_cost: Decimal | None = Field(default=None, ge=Decimal("0"), description="Known combined transport cost.")
    activity_cost: Decimal | None = Field(default=None, ge=Decimal("0"), description="Known activity cost.")
    miscellaneous_cost: Decimal | None = Field(default=None, ge=Decimal("0"), description="Known other expenses.")
    total_estimated_cost: Decimal | None = Field(
        default=None, ge=Decimal("0"), description="Known subtotal plus explicit emergency buffer."
    )
    currency: str = Field(default="INR", min_length=3, max_length=3, description="Currency code.")
    remaining_budget: Decimal | None = Field(
        default=None, description="Remaining budget; provisional when the estimate is partial."
    )
    local_transportation_cost: Decimal | None = Field(default=None, ge=Decimal("0"), description="Known local transportation cost.")
    intercity_transportation_cost: Decimal | None = Field(default=None, ge=Decimal("0"), description="Known intercity transportation cost.")
    line_items: list[BudgetLineItem] = Field(default_factory=list, description="Auditable source price calculations.")
    known_subtotal: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"), description="Sum of included known expenses before buffer.")
    emergency_buffer: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0"), description="Explicitly configured emergency buffer amount.")
    emergency_buffer_percent: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="Configured emergency buffer percentage; default zero is shown explicitly.")
    estimated_cost_per_person: Decimal | None = Field(default=None, ge=Decimal("0"), description="Estimated total divided by travelers.")
    user_budget: Decimal | None = Field(default=None, ge=Decimal("0"), description="User's total trip budget, if supplied.")
    budget_utilization_percent: Decimal | None = Field(default=None, ge=Decimal("0"), description="Budget utilization percentage; provisional for partial estimates.")
    missing_categories: list[BudgetCategory] = Field(default_factory=list, description="Categories lacking reliable prices.")
    estimate_status: Literal["complete", "partial", "assumption_based", "partial_with_assumptions"] = Field(default="partial", description="Completeness and approved-assumption state.")
    budget_comparison_status: Literal["within_limit", "over_budget", "provisional", "not_available"] = Field(default="not_available", description="Budget comparison state.")

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize a three-character currency code to uppercase."""
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("Currency must be a three-letter code.")
        return value

    @model_validator(mode="after")
    def validate_totals(self) -> BudgetBreakdown:
        """Check category subtotals where all legacy components are available."""
        components = (
            self.accommodation_cost,
            self.food_cost,
            self.transportation_cost,
            self.activity_cost,
            self.miscellaneous_cost,
        )
        if all(component is not None for component in components):
            component_total = sum((component for component in components if component is not None), Decimal("0"))
            if "known_subtotal" in self.model_fields_set and abs(
                component_total - self.known_subtotal
            ) > Decimal("0.01"):
                raise ValueError("Known subtotal must match the supplied category costs.")
            if (
                "known_subtotal" not in self.model_fields_set
                and self.total_estimated_cost is not None
                and abs(component_total - self.total_estimated_cost) > Decimal("0.01")
            ):
                raise ValueError(
                    "Total estimated cost must equal the sum of all supplied cost components."
                )
        if self.total_estimated_cost is not None and self.total_estimated_cost < self.known_subtotal:
            raise ValueError("Final estimated cost cannot be below the known subtotal.")
        return self


InformationStatus = Literal["verified", "general_guidance", "unavailable"]
AdviceBasis = Literal["verified_source", "forecast", "itinerary", "general_guidance"]


class InformationSource(SchemaModel):
    """External or explicitly general source attribution for travel information."""

    title: str | None = Field(default=None, description="Source page or provider title.")
    url: str | None = Field(default=None, description="Source URL, when available.")
    provider: str | None = Field(default=None, description="Provider or organization name.")
    retrieved_at: str | None = Field(default=None, description="UTC retrieval timestamp, if retrieved.")
    information_type: Literal["forecast", "web_research", "general_guidance"] = Field(
        ..., description="Whether this is a forecast, cited web research, or generic guidance."
    )


class DailyWeather(SchemaModel):
    """One provider-returned daily forecast observation, not a climate average."""

    date: DateType = Field(..., description="Forecast calendar date.")
    condition: str | None = Field(default=None, description="Provider-described forecast condition.")
    temperature_min_c: float | None = Field(default=None, description="Forecast minimum temperature in Celsius.")
    temperature_max_c: float | None = Field(default=None, description="Forecast maximum temperature in Celsius.")
    precipitation_mm: float | None = Field(default=None, ge=0, description="Forecast daily precipitation sum in millimeters.")
    source: InformationSource = Field(..., description="Weather provider and retrieval information.")


class WeatherReport(SchemaModel):
    """Forecast coverage and unavailable dates for the requested trip window."""

    status: Literal["forecast", "partial_forecast", "unavailable"] = Field(
        ..., description="Forecast coverage for the requested travel dates."
    )
    location: str | None = Field(default=None, description="Geocoded forecast location label.")
    provider: str = Field(default="Open-Meteo", description="Forecast provider.")
    retrieved_at: str | None = Field(default=None, description="UTC retrieval timestamp.")
    daily_forecasts: list[DailyWeather] = Field(default_factory=list, description="Actual provider forecasts only.")
    unavailable_dates: list[DateType] = Field(default_factory=list, description="Trip dates without a provider forecast.")
    unavailable_reason: str | None = Field(default=None, description="Why the forecast is unavailable or partial.")
    recommendations: list[str] = Field(default_factory=list, description="Weather recommendations derived only from returned forecasts.")


class AdviceItem(SchemaModel):
    """A travel recommendation with an explicit basis and source references."""

    text: str = Field(..., min_length=1, description="Practical recommendation.")
    basis: AdviceBasis = Field(..., description="Forecast, itinerary, verified source, or generic basis.")
    sources: list[InformationSource] = Field(default_factory=list, description="Sources supporting this recommendation.")


class TransportationInformation(SchemaModel):
    """Transport guidance separated by whether it is source-backed or general."""

    status: InformationStatus = Field(..., description="Evidence status for transport information.")
    local_options: list[AdviceItem] = Field(default_factory=list, description="Local/public transport and taxi guidance.")
    intercity_options: list[AdviceItem] = Field(default_factory=list, description="Intercity transport guidance.")
    logistics_tips: list[AdviceItem] = Field(default_factory=list, description="General access and trip logistics guidance.")


class PackingRecommendations(SchemaModel):
    """Packing suggestions grouped into the requested practical categories."""

    clothing: list[AdviceItem] = Field(default_factory=list)
    footwear: list[AdviceItem] = Field(default_factory=list)
    personal_essentials: list[AdviceItem] = Field(default_factory=list)
    travel_documents: list[AdviceItem] = Field(default_factory=list)
    electronics_accessories: list[AdviceItem] = Field(default_factory=list)
    weather_specific: list[AdviceItem] = Field(default_factory=list)
    activity_specific: list[AdviceItem] = Field(default_factory=list)
    forecast_based: bool = Field(default=False, description="True only if weather-specific items use a returned forecast.")


class TravelAssistantDraft(SchemaModel):
    """LLM-organized advice payload; deliberately contains no weather readings."""

    transportation: TransportationInformation = Field(..., description="Transport guidance grounded in supplied research.")
    packing: PackingRecommendations = Field(..., description="Categorized packing suggestions.")
    practical_tips: list[AdviceItem] = Field(default_factory=list, description="Cited or explicitly general practical tips.")


class TravelInformation(SchemaModel):
    """Structured weather, transport, packing, and practical trip information."""

    weather_summary: str | None = Field(default=None, description="Human-readable forecast availability summary.")
    temperature: str | None = Field(default=None, description="Forecast temperature range when available.")
    transport_options: list[str] = Field(default_factory=list, description="Backward-compatible transport strings.")
    accommodation_suggestions: list[str] = Field(default_factory=list, description="Accommodation suggestions, when researched.")
    packing_recommendations: list[str] = Field(default_factory=list, description="Backward-compatible packing strings.")
    safety_tips: list[str] = Field(default_factory=list, description="Backward-compatible safety strings.")
    weather: WeatherReport | None = Field(default=None, description="Structured forecast and coverage data.")
    transportation: TransportationInformation | None = Field(default=None, description="Structured logistics and source status.")
    packing: PackingRecommendations | None = Field(default=None, description="Categorized packing suggestions.")
    practical_tips: list[AdviceItem] = Field(default_factory=list, description="Source-aware practical tips.")
    sources: list[InformationSource] = Field(default_factory=list, description="Deduplicated sources consulted.")


ValidationSeverity = Literal["PASS", "WARNING", "ERROR"]
ValidationCategory = Literal[
    "travel_request", "destination_research", "itinerary", "budget", "travel_information"
]
ValidationStatus = Literal["PASSED", "PASSED_WITH_WARNINGS", "FAILED", "PARTIAL"]


class ValidationIssue(SchemaModel):
    """One structured result from a deterministic validation rule."""

    severity: ValidationSeverity = Field(..., description="PASS, WARNING, or ERROR.")
    category: ValidationCategory = Field(..., description="Plan section evaluated by the check.")
    code: str = Field(..., min_length=1, description="Stable rule identifier.")
    message: str = Field(..., min_length=1, description="Human-readable check result.")
    related_field: str | None = Field(default=None, description="State/model field related to the issue.")


class ValidationResult(SchemaModel):
    """Structured summary of deterministic travel-plan validation."""

    status: ValidationStatus = Field(..., description="Overall validation status.")
    is_valid: bool = Field(..., description="False only when one or more ERROR issues exist.")
    checks_performed: int = Field(default=0, ge=0, description="Number of individual checks recorded.")
    checks_passed: int = Field(default=0, ge=0, description="Number of PASS checks.")
    error_count: int = Field(default=0, ge=0, description="Number of ERROR issues.")
    warning_count: int = Field(default=0, ge=0, description="Number of WARNING issues.")
    issues: list[ValidationIssue] = Field(default_factory=list, description="Structured check results.")
    suggestions: list[str] = Field(default_factory=list, description="Non-destructive follow-up suggestions.")
    budget_within_limit: bool | None = Field(
        default=None, description="Whether the independently checked estimate fits the supplied budget."
    )
    itinerary_feasible: bool | None = Field(
        default=None, description="Whether basic itinerary structure and date coverage checks passed."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_report(cls, value: object) -> object:
        """Accept the earlier string-issue report shape and upgrade it in memory."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw_issues = data.get("issues", [])
        if isinstance(raw_issues, list):
            legacy_invalid = data.get("is_valid") is False
            issues = [
                {
                    "severity": "ERROR" if legacy_invalid else "WARNING",
                    "category": "travel_request",
                    "code": "legacy_issue",
                    "message": issue,
                }
                if isinstance(issue, str)
                else issue
                for issue in raw_issues
            ]
            data["issues"] = issues
        else:
            issues = []

        def severity_of(issue: object) -> object:
            if isinstance(issue, dict):
                return issue.get("severity")
            return getattr(issue, "severity", None)

        if data.get("is_valid") is False and not any(
            severity_of(issue) == "ERROR" for issue in issues
        ):
            issues.append(
                {
                    "severity": "ERROR",
                    "category": "travel_request",
                    "code": "legacy_invalid_result",
                    "message": "Legacy validation result was marked invalid.",
                }
            )
            data["issues"] = issues
        errors = sum(severity_of(issue) == "ERROR" for issue in issues)
        warnings = sum(severity_of(issue) == "WARNING" for issue in issues)
        passed = sum(severity_of(issue) == "PASS" for issue in issues)
        data.setdefault("checks_performed", len(issues))
        data.setdefault("checks_passed", passed)
        data.setdefault("error_count", errors)
        data.setdefault("warning_count", warnings)
        data.setdefault(
            "status",
            "FAILED" if errors else "PASSED_WITH_WARNINGS" if warnings else "PASSED",
        )
        return data

    @model_validator(mode="after")
    def counts_match_issues(self) -> ValidationResult:
        """Keep summary counters consistent with the structured check records."""
        errors = sum(issue.severity == "ERROR" for issue in self.issues)
        warnings = sum(issue.severity == "WARNING" for issue in self.issues)
        passed = sum(issue.severity == "PASS" for issue in self.issues)
        if self.checks_performed != len(self.issues):
            raise ValueError("checks_performed must equal the number of validation issues.")
        if (self.error_count, self.warning_count, self.checks_passed) != (errors, warnings, passed):
            raise ValueError("Validation counters must match issue severities.")
        if self.is_valid != (errors == 0):
            raise ValueError("is_valid must be false exactly when ERROR issues exist.")
        if errors and self.status != "FAILED":
            raise ValueError("Validation status must be FAILED when ERROR issues exist.")
        if not errors and warnings and self.status == "PASSED":
            raise ValueError("Validation status cannot be PASSED while WARNING issues exist.")
        if not errors and not warnings and self.status == "PASSED_WITH_WARNINGS":
            raise ValueError("PASSED_WITH_WARNINGS requires at least one WARNING issue.")
        if not errors and not warnings and self.status == "PARTIAL":
            raise ValueError("PARTIAL status requires at least one WARNING issue.")
        return self


class FinalTravelPlan(SchemaModel):
    """Final structured plan that can be serialized for an API or UI."""

    destination: str | None = Field(default=None, description="Planned destination, when available.")
    trip_summary: str = Field(default="", description="Summary based on available trip information.")
    itinerary: list[ItineraryDay] = Field(default_factory=list, description="Day-by-day plan.")
    destination_research: DestinationResearch | None = Field(
        default=None, description="Validated destination research, if available."
    )
    budget_breakdown: BudgetBreakdown | None = Field(
        default=None, description="Budget estimate, if available."
    )
    travel_information: TravelInformation | None = Field(
        default=None, description="Practical travel information, if available."
    )
    important_tips: list[str] = Field(default_factory=list, description="Important traveler tips.")
    total_estimated_cost: Decimal | None = Field(
        default=None, ge=0, description="Total estimated trip cost, if available."
    )
    currency: str = Field(default="INR", min_length=3, max_length=3, description="Currency code.")
    generation_status: Literal["complete", "partial"] = Field(
        default="partial", description="Whether every expected upstream output was available."
    )
    markdown_response: str = Field(default="", description="Deterministically formatted Markdown suitable for display.")
    missing_information: list[str] = Field(default_factory=list, description="Unavailable or incomplete upstream information.")
    validation_result: ValidationResult | None = Field(default=None, description="Exact validation report, when available.")
    source_urls: list[str] = Field(default_factory=list, description="Source URLs copied unchanged from upstream outputs.")

    @field_validator("destination")
    @classmethod
    def destination_must_not_be_blank(cls, value: str | None) -> str | None:
        """Reject empty or whitespace-only destinations."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Destination must not be empty.")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Normalize a three-character currency code to uppercase."""
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("Currency must be a three-letter code.")
        return value
