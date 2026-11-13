"""
backend/api/
=============
FastAPI route handlers and middleware for the AI Travel Planner.

Routers planned for Phase 6+:
    - travel.py     : POST /api/v1/travel/plan – submit a travel planning request.
    - sessions.py   : GET/DELETE /api/v1/sessions – manage planning sessions.
    - health.py     : Additional detailed health-check endpoints.

Routers are registered in backend/main.py via app.include_router().
"""
