"""
src/workflow/graph.py

LangGraph StateGraph orchestration for the AI Finance Assistant.
Coordinates the workflow lifecycle:
  START -> router_node -> [qa | portfolio | market | goals | news | tax] -> merger_node -> END
Maintains multi-turn context, user profile persistence, and structured outputs.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.goal_agent import GoalPlanningAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.news_agent import NewsSynthesizerAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent, parse_holdings_from_text
from src.agents.qa_agent import FinanceQAAgent
from src.agents.tax_agent import TaxEducationAgent
from src.core.logging_config import get_logger
from src.core.state import AppState
from src.workflow.router import route_intent

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy-initialized agent singletons
# ---------------------------------------------------------------------------
class AgentRegistry:
    def __init__(self) -> None:
        self._qa: FinanceQAAgent | None = None
        self._portfolio: PortfolioAnalysisAgent | None = None
        self._market: MarketAnalysisAgent | None = None
        self._goals: GoalPlanningAgent | None = None
        self._news: NewsSynthesizerAgent | None = None
        self._tax: TaxEducationAgent | None = None

    @property
    def qa(self) -> FinanceQAAgent:
        if self._qa is None:
            self._qa = FinanceQAAgent()
        return self._qa

    @property
    def portfolio(self) -> PortfolioAnalysisAgent:
        if self._portfolio is None:
            self._portfolio = PortfolioAnalysisAgent()
        return self._portfolio

    @property
    def market(self) -> MarketAnalysisAgent:
        if self._market is None:
            self._market = MarketAnalysisAgent()
        return self._market

    @property
    def goals(self) -> GoalPlanningAgent:
        if self._goals is None:
            self._goals = GoalPlanningAgent()
        return self._goals

    @property
    def news(self) -> NewsSynthesizerAgent:
        if self._news is None:
            self._news = NewsSynthesizerAgent()
        return self._news

    @property
    def tax(self) -> TaxEducationAgent:
        if self._tax is None:
            self._tax = TaxEducationAgent()
        return self._tax


registry = AgentRegistry()


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------


def router_node(state: AppState) -> dict[str, Any]:
    """Classify the latest user message and determine which agent should handle it."""
    messages = state.get("messages", [])
    last_user_text = ""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human" or isinstance(msg, HumanMessage):
            last_user_text = getattr(msg, "content", str(msg))
            break

    intent = route_intent(last_user_text, messages)
    logger.info("Routing user message", intent=intent, query=last_user_text[:60])
    return {"intent": intent}


def qa_node(state: AppState) -> dict[str, Any]:
    output = registry.qa._safe_invoke(state)
    return {"agent_outputs": [output]}


def portfolio_node(state: AppState) -> dict[str, Any]:
    # Check if we should update portfolio holdings in state from user text
    messages = state.get("messages", [])
    last_text = getattr(messages[-1], "content", "") if messages else ""
    parsed = parse_holdings_from_text(last_text)

    output = registry.portfolio._safe_invoke(state)

    updates: dict[str, Any] = {"agent_outputs": [output]}
    if parsed:
        existing = list(state.get("portfolio", []))
        # Add new holdings or update existing
        existing.extend(parsed)
        updates["portfolio"] = existing

    return updates


def market_node(state: AppState) -> dict[str, Any]:
    output = registry.market._safe_invoke(state)
    return {"agent_outputs": [output]}


def goals_node(state: AppState) -> dict[str, Any]:
    output = registry.goals._safe_invoke(state)
    return {"agent_outputs": [output]}


def news_node(state: AppState) -> dict[str, Any]:
    output = registry.news._safe_invoke(state)
    return {"agent_outputs": [output]}


def tax_node(state: AppState) -> dict[str, Any]:
    output = registry.tax._safe_invoke(state)
    return {"agent_outputs": [output]}


def merger_node(state: AppState) -> dict[str, Any]:
    """
    Consolidate agent outputs into final_response, format citations,
    and append the AIMessage to conversation history.
    """
    outputs = state.get("agent_outputs", [])

    if not outputs:
        fallback_msg = (
            "I'm here to help with your financial questions. What would you like to explore?"
        )
        return {
            "final_response": fallback_msg,
            "messages": [AIMessage(content=fallback_msg)],
        }

    # Format final merged response
    combined_content = "\n\n".join(
        [out.get("content", "") for out in outputs if out.get("content")]
    )

    # Append AIMessage with metadata
    ai_msg = AIMessage(
        content=combined_content,
        additional_kwargs={
            "agent_name": outputs[0].get("agent_name", "AI Finance Assistant"),
            "intent": state.get("intent", "qa"),
            "sources": outputs[0].get("sources", []),
            "data": outputs[0].get("data", {}),
        },
    )

    return {
        "final_response": combined_content,
        "messages": [ai_msg],
    }


def route_decision(state: AppState) -> str:
    """Conditional routing edge deciding which node to visit next."""
    intent = state.get("intent", "qa")
    valid_nodes = {"qa", "portfolio", "market", "goals", "news", "tax"}
    return intent if intent in valid_nodes else "qa"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------


def create_finance_graph(checkpointer: Any | None = None) -> Any:
    """
    Construct and compile the LangGraph workflow.

    Args:
        checkpointer: Optional MemorySaver or external persistence checkpointer.
                      Defaults to an in-memory MemorySaver.
    """
    builder = StateGraph(AppState)

    # 1. Add all nodes
    builder.add_node("router_node", router_node)
    builder.add_node("qa", qa_node)
    builder.add_node("portfolio", portfolio_node)
    builder.add_node("market", market_node)
    builder.add_node("goals", goals_node)
    builder.add_node("news", news_node)
    builder.add_node("tax", tax_node)
    builder.add_node("merger_node", merger_node)

    # 2. Add start edge to router
    builder.add_edge(START, "router_node")

    # 3. Add conditional edge from router to agents
    builder.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "qa": "qa",
            "portfolio": "portfolio",
            "market": "market",
            "goals": "goals",
            "news": "news",
            "tax": "tax",
        },
    )

    # 4. Add return edges from all agents to merger_node
    builder.add_edge("qa", "merger_node")
    builder.add_edge("portfolio", "merger_node")
    builder.add_edge("market", "merger_node")
    builder.add_edge("goals", "merger_node")
    builder.add_edge("news", "merger_node")
    builder.add_edge("tax", "merger_node")

    # 5. Add finish edge
    builder.add_edge("merger_node", END)

    # 6. Compile with checkpointer for state memory
    memory = checkpointer if checkpointer is not None else MemorySaver()
    compiled = builder.compile(checkpointer=memory)

    logger.info("Compiled LangGraph finance workflow graph successfully")
    return compiled


# Global compiled graph instance
default_graph = create_finance_graph()


def run_finance_turn(
    user_input: str,
    thread_id: str = "default_session",
    graph: Any | None = None,
    user_profile: dict | None = None,
    portfolio: list | None = None,
) -> AppState:
    """
    Convenience function to run a single conversation turn through the graph.

    Args:
        user_input: The user's message string.
        thread_id: Session identifier for multi-turn checkpointer state.
        graph: Optional compiled graph instance.
        user_profile: Optional user profile data to inject.
        portfolio: Optional portfolio holdings list to inject.

    Returns:
        The updated AppState after execution.
    """
    g = graph or default_graph
    config = {"configurable": {"thread_id": thread_id}}

    inputs: dict[str, Any] = {
        "messages": [HumanMessage(content=user_input)],
    }
    if user_profile is not None:
        inputs["user_profile"] = user_profile
    if portfolio is not None:
        inputs["portfolio"] = portfolio

    result_state = g.invoke(inputs, config=config)
    return result_state
