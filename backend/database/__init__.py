"""
backend/database/
==================
SQLAlchemy database configuration, models, and session management.

Modules planned for Phase 8+:
    - session.py        : Async SQLAlchemy engine and session factory.
    - base.py           : Declarative Base for ORM models.
    - models/
        - travel_plan.py    : Persisted travel plan records.
        - user_session.py   : User session and preferences storage.

The database URL is controlled by the DATABASE_URL environment variable
(default: SQLite at ./travel_planner.db).
"""
