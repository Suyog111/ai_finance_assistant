"""
src/agents/base_agent.py

Abstract base class all specialized agents inherit from.
Enforces a consistent interface: invoke(state) → AgentOutput.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.logging_config import get_logger
from src.core.state import AgentOutput, AppState


class BaseFinanceAgent(ABC):
    """
    Every specialized agent must implement `invoke`.

    Attributes:
        name:        Human-readable agent name (used in logs and UI badges).
        description: Short description shown in UI / routing prompts.
    """

    name: str = "base"
    description: str = "Base agent — not for direct use."

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def invoke(self, state: AppState) -> AgentOutput:
        """
        Process the current conversation state and return an AgentOutput.

        Args:
            state: The current LangGraph AppState.

        Returns:
            AgentOutput with at minimum `agent_name` and `content`.
        """
        ...

    def _safe_invoke(self, state: AppState) -> AgentOutput:
        """Wraps invoke() with error handling — called by the LangGraph node."""
        try:
            return self.invoke(state)
        except Exception as exc:
            self.logger.error("Agent error", agent=self.name, error=str(exc), exc_info=True)
            return AgentOutput(
                agent_name=self.name,
                content=(
                    "I'm sorry, I encountered an issue processing your request. "
                    "Please try again or rephrase your question."
                ),
                sources=[],
                data={},
                error=str(exc),
            )
