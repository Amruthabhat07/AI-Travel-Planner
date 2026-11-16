"""
backend/agents/
================
Specialist AI agent implementations for the AI Travel Planner.

Implemented:
    - DestinationResearchAgent: Tavily-backed destination research with Gemini.
    - ItineraryPlanningAgent: Gemini itinerary generation with geographic routing.
    - BudgetPlanningAgent: Deterministic Decimal-based expense calculations.
    - TravelAssistantAgent: Source-aware weather, logistics, packing, and travel advice.
    - ValidationAgent: Python/Pydantic structural and financial consistency checks.
    - FinalResponseGenerator: Python-only structured and Markdown final plan formatting.

Planned:
    - CoordinatorAgent   : Understands user intent and routes tasks.
    - ItineraryAgent     : Generates day-by-day travel itineraries.

Agents communicate through the shared state defined in backend/graph/.
"""
