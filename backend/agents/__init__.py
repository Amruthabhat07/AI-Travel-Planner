"""
backend/agents/
================
Specialist AI agent implementations for the AI Travel Planner.

Implemented:
    - DestinationResearchAgent: Tavily-backed destination research with Gemini.
    - ItineraryPlanningAgent: Gemini itinerary generation with geographic routing.
    - BudgetPlanningAgent: Deterministic Decimal-based expense calculations.
    - TravelAssistantAgent: Source-aware weather, logistics, packing, and travel advice.

Planned:
    - CoordinatorAgent   : Understands user intent and routes tasks.
    - ItineraryAgent     : Generates day-by-day travel itineraries.
    - ValidationAgent    : Reviews the plan for consistency and feasibility.
    - ResponseGenerator  : Compiles all outputs into the final travel guide.

Agents communicate through the shared state defined in backend/graph/.
"""
