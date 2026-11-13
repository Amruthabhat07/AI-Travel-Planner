"""
backend/graph/
===============
LangGraph state graph and workflow orchestration for the AI Travel Planner.

This package will contain:
    - state.py        : TypedDict / Pydantic definition of the shared agent state.
    - workflow.py     : LangGraph StateGraph construction and compilation.
    - nodes.py        : Node wrapper functions that call individual agents.
    - edges.py        : Conditional edge logic for routing between agents.

Implementation starts in Phase 2.
"""
