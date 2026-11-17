"""
src/agents/qa_agent.py

Finance Q&A Agent.
Handles general financial education queries using RAG retrieval.
Explains personal finance, investing fundamentals, concepts, and terminology.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseFinanceAgent
from src.core.llm_client import get_llm_for_agent
from src.core.state import AgentOutput, AppState
from src.rag.rag_chain import RAGChain
from src.rag.retriever import FinanceRetriever
from src.rag.vector_store import get_or_build_vector_store
from src.utils.disclaimers import GENERAL_DISCLAIMER


class FinanceQAAgent(BaseFinanceAgent):
    """Specialized agent for general financial literacy and investment education."""

    name: str = "Finance Q&A Agent"
    description: str = "Explains investing basics, financial concepts, and personal finance terms."

    def __init__(self, vector_store: Any | None = None) -> None:
        super().__init__()
        try:
            store = vector_store or get_or_build_vector_store()
            self.retriever = FinanceRetriever(store)
        except Exception as exc:
            self.logger.warning("Could not initialize vector store in QA Agent", error=str(exc))
            self.retriever = None  # type: ignore[assignment]

        self.llm = get_llm_for_agent("qa_agent", temperature=0.1)
        self.chain = RAGChain(self.retriever, self.llm) if self.retriever else None

    def invoke(self, state: AppState) -> AgentOutput:
        """Process user query and return educational response grounded in RAG."""
        # Extract latest user message
        messages = state.get("messages", [])
        if not messages:
            return AgentOutput(
                agent_name=self.name,
                content="Hello! How can I help you learn about investing and personal finance today?",
                sources=[],
                data={},
            )

        last_message = messages[-1]
        user_query = getattr(last_message, "content", str(last_message))

        # Check if user profile has knowledge level to adapt response
        profile = state.get("user_profile", {})
        knowledge_level = profile.get("knowledge_level", "beginner")

        # Format conversation history
        conv_history = []
        for msg in messages[:-1]:
            role = "user" if getattr(msg, "type", "") == "human" else "assistant"
            conv_history.append({"role": role, "content": getattr(msg, "content", str(msg))})

        if self.chain:
            # Query RAG chain (across basics and general knowledge)
            rag_response = self.chain.invoke(
                question=user_query,
                category=None,  # Search all categories
                conversation_history=conv_history,
            )
            content = rag_response.answer + GENERAL_DISCLAIMER
            sources = rag_response.sources
        else:
            # Fallback if vector store is unavailable
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = (
                f"You are a helpful financial educator. Explain this concept clearly for a {knowledge_level} level:\n"
                f"{user_query}"
            )
            response = self.llm.invoke(
                [SystemMessage(content=prompt), HumanMessage(content=user_query)]
            )
            content = str(response.content) + GENERAL_DISCLAIMER
            sources = []

        return AgentOutput(
            agent_name=self.name,
            content=content,
            sources=sources,
            data={"knowledge_level": knowledge_level},
            error=None,
        )
