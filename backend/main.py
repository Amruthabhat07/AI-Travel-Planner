"""
backend/main.py
================
FastAPI application entry point for the AI Travel Planner backend.

Phase 11:
    - Application factory with lifespan context manager.
    - Root health-check endpoint.
    - Structured logging configuration.
    - LangSmith tracing initialisation hook.

    - REST routes invoke the existing LangGraph workflow.
    - Database integration remains a future phase.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown events.

    On startup:
        - Load and validate configuration.
        - Initialise LangSmith tracing (if enabled).
        - Log readiness.

    On shutdown:
        - Perform graceful cleanup (database connections, etc. in future phases).
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("AI Travel Planner backend starting up …")

    cfg = get_settings()
    logging.getLogger().setLevel(cfg.log_level.upper())

    # Propagate LangSmith credentials to the environment.
    cfg.configure_langsmith()

    logger.info(
        "Configuration loaded. environment=%s | tracing=%s",
        cfg.app_env,
        cfg.langsmith_tracing,
    )
    logger.info("Backend is ready. Listening on %s:%d", cfg.api_host, cfg.api_port)

    yield  # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("AI Travel Planner backend shutting down. Goodbye!")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Returns:
        FastAPI: The configured application object.
    """
    cfg = get_settings()

    application = FastAPI(
        title="AI Travel Planner API",
        description=(
            "Multi-agent AI backend for personalised travel planning. "
            "Powered by LangGraph, LangChain, and Google Gemini."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Origins are explicit in every environment; production never defaults to '*'.
    origins = cfg.allowed_cors_origins

    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────────
    _register_routes(application)

    return application


def _register_routes(app: FastAPI) -> None:
    """
    Register all API routes on the given application.

    Register lightweight operational routes and the Phase 11 travel router.
    """

    @app.get("/", tags=["Root"], summary="API root")
    async def root() -> JSONResponse:
        """Return a welcome message and links to the API documentation."""
        return JSONResponse(
            content={
                "name": "AI Travel Planner API",
                "version": "0.1.0",
                "description": "REST API for the LangGraph multi-agent travel planner.",
                "phase": "Phase 1 – Project Foundation",
                "docs": "/docs",
                "health": "/health",
            }
        )

    @app.get("/health", tags=["Health"], summary="Health check")
    async def health_check() -> JSONResponse:
        """
        Lightweight liveness probe.

        Returns HTTP 200 with a JSON body when the application is running.
        Extend this in later phases to check database connectivity and agent
        availability.
        """
        return JSONResponse(
            content={
                "status": "healthy",
                "application": "AI Travel Planner API",
                "operational": True,
                "version": "0.1.0",
            }
        )

    from backend.api.travel import router as travel_router

    app.include_router(travel_router)


# ---------------------------------------------------------------------------
# Application instance (used by uvicorn and import statements)
# ---------------------------------------------------------------------------

app = create_app()

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=cfg.is_development,
        log_level=cfg.log_level.lower(),
    )
