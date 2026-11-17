"""
src/core/state.py

Shared LangGraph state definitions.
All agents read from and write to these TypedDicts.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# User Profile  (persisted across turns)
# ---------------------------------------------------------------------------


class UserProfile(TypedDict, total=False):
    """Optional user profile data collected during conversation."""

    name: str
    risk_tolerance: str  # "conservative" | "moderate" | "aggressive"
    investment_horizon: str  # e.g. "5 years", "20 years"
    monthly_income: float
    stated_goals: list[str]  # e.g. ["retirement", "emergency fund"]
    knowledge_level: str  # "beginner" | "intermediate" | "advanced"


# ---------------------------------------------------------------------------
# Portfolio Holding  (input to Portfolio Analysis Agent)
# ---------------------------------------------------------------------------


class Holding(TypedDict):
    ticker: str
    quantity: float
    avg_cost: float  # Average purchase price per share


# ---------------------------------------------------------------------------
# Agent Output  (structured response from any agent)
# ---------------------------------------------------------------------------


class AgentOutput(TypedDict, total=False):
    agent_name: str
    content: str  # Human-readable answer
    sources: list[dict[str, str]]  # [{title, url, category}]
    data: dict[str, Any]  # Structured data (metrics, charts, etc.)
    error: str | None


# ---------------------------------------------------------------------------
# Main Application State  (LangGraph state)
# ---------------------------------------------------------------------------


class AppState(TypedDict):
    """
    Central state object flowing through the LangGraph graph.

    - `messages`:        Full conversation history (auto-merged by LangGraph).
    - `user_profile`:    Incrementally-built user profile.
    - `portfolio`:       Current user holdings.
    - `intent`:          Classified routing intent (set by router node).
    - `agent_outputs`:   Outputs from one or more agents in this turn.
    - `final_response`:  Merged / formatted reply sent to the user.
    - `error`:           Error message if something failed this turn.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: UserProfile
    portfolio: list[Holding]
    intent: str  # "qa" | "portfolio" | "market" | "goals" | "news" | "tax" | "multi"
    agent_outputs: list[AgentOutput]
    final_response: str
    error: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def initial_state() -> AppState:
    """Return a blank AppState for a new session."""
    return AppState(
        messages=[],
        user_profile={},
        portfolio=[],
        intent="",
        agent_outputs=[],
        final_response="",
        error=None,
    )
