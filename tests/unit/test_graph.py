"""
tests/unit/test_graph.py

Unit tests for src/workflow/graph.py
Tests node execution, routing decisions, and graph compilation.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.core.state import AgentOutput, initial_state
from src.workflow.graph import (
    create_finance_graph,
    merger_node,
    route_decision,
    router_node,
)


class TestGraphNodes:
    def test_router_node_sets_intent(self):
        state = initial_state()
        state["messages"] = [HumanMessage(content="What is a stock?")]

        with patch("src.workflow.graph.route_intent", return_value="qa"):
            update = router_node(state)
            assert update["intent"] == "qa"

    def test_route_decision(self):
        state = initial_state()
        state["intent"] = "tax"
        assert route_decision(state) == "tax"

        state["intent"] = "invalid_node"
        assert route_decision(state) == "qa"

    def test_merger_node_consolidates_outputs(self):
        state = initial_state()
        state["intent"] = "qa"
        state["agent_outputs"] = [
            AgentOutput(
                agent_name="Finance Q&A Agent",
                content="A stock represents partial ownership in a corporation.",
                sources=[{"title": "Stocks Explained", "category": "basics"}],
                data={},
            )
        ]

        update = merger_node(state)
        assert "A stock represents partial ownership" in update["final_response"]
        assert len(update["messages"]) == 1
        assert update["messages"][0].content == update["final_response"]

    def test_merger_node_empty_outputs_fallback(self):
        state = initial_state()
        state["agent_outputs"] = []
        update = merger_node(state)
        assert "I'm here to help" in update["final_response"]


class TestGraphCompilation:
    def test_create_finance_graph_compiles(self):
        graph = create_finance_graph()
        assert graph is not None

    def test_graph_execution_mocked_end_to_end(self):
        graph = create_finance_graph()

        mock_out = AgentOutput(
            agent_name="Mock QA Agent",
            content="Educational financial explanation.",
            sources=[],
            data={},
        )

        with (
            patch("src.workflow.graph.registry.qa._safe_invoke", return_value=mock_out),
            patch("src.workflow.graph.route_intent", return_value="qa"),
        ):
            config = {"configurable": {"thread_id": "test_thread_1"}}
            inputs = {"messages": [HumanMessage(content="Explain compound interest")]}
            result = graph.invoke(inputs, config=config)

            assert result["intent"] == "qa"
            assert "Educational financial explanation." in result["final_response"]
            # Conversation messages should have 1 Human + 1 AI
            assert len(result["messages"]) >= 2
