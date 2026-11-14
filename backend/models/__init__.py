"""
backend/models/
================
Pydantic data models and shared state definitions for the AI Travel Planner.

Reusable Pydantic v2 schemas for validated travel requests and agent outputs.
"""

from backend.models.schemas import (
    Attraction,
    BudgetBreakdown,
    BudgetLineItem,
    BudgetPriceInput,
    DestinationResearch,
    DailyWeather,
    InformationSource,
    FinalTravelPlan,
    ItineraryDay,
    PackingRecommendations,
    TransportationInformation,
    WeatherReport,
    AdviceItem,
    TravelInformation,
    TravelRequest,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    "Attraction",
    "BudgetBreakdown",
    "BudgetLineItem",
    "BudgetPriceInput",
    "DestinationResearch",
    "DailyWeather",
    "InformationSource",
    "FinalTravelPlan",
    "ItineraryDay",
    "PackingRecommendations",
    "TransportationInformation",
    "WeatherReport",
    "AdviceItem",
    "TravelInformation",
    "TravelRequest",
    "ValidationIssue",
    "ValidationResult",
]
