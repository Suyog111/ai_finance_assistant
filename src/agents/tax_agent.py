"""
src/agents/tax_agent.py

Tax Education Agent.
Explains tax concepts, tax-advantaged accounts (US 401k/IRA/HSA and Indian 80C/ELSS/PPF/NPS),
capital gains rules (IRS brackets & Indian Union Budget 2024 rates), and tax-efficient investing.
Always appends mandatory tax disclaimers.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseFinanceAgent
from src.core.llm_client import get_llm_for_agent
from src.core.state import AgentOutput, AppState
from src.rag.rag_chain import RAGChain
from src.rag.retriever import FinanceRetriever
from src.rag.vector_store import get_or_build_vector_store
from src.utils.disclaimers import TAX_DISCLAIMER


class TaxEducationAgent(BaseFinanceAgent):
    """Specialized agent for explaining tax rules, deductions, and account structures."""

    name: str = "Tax Education Agent"
    description: str = (
        "Explains capital gains taxes, tax-advantaged accounts, deductions, and tax efficiency."
    )

    def __init__(self, vector_store: Any | None = None) -> None:
        super().__init__()
        self.llm = get_llm_for_agent("tax_agent", temperature=0.1)

        try:
            store = vector_store or get_or_build_vector_store()
            self.retriever = FinanceRetriever(store)
            self.chain = RAGChain(self.retriever, self.llm)
        except Exception as exc:
            self.logger.warning("Could not initialize vector store in Tax Agent", error=str(exc))
            self.retriever = None  # type: ignore[assignment]
            self.chain = None

    def invoke(self, state: AppState) -> AgentOutput:
        """Answer tax query grounded in verified tax articles and knowledge base."""
        messages = state.get("messages", [])
        if not messages:
            return AgentOutput(
                agent_name=self.name,
                content="Hello! I can explain tax concepts such as capital gains, tax-advantaged accounts, and deductions.",
                sources=[],
                data={},
            )

        last_message = messages[-1]
        user_query = getattr(last_message, "content", str(last_message))

        # Build conversation history
        conv_history = []
        for msg in messages[:-1]:
            role = "user" if getattr(msg, "type", "") == "human" else "assistant"
            conv_history.append({"role": role, "content": getattr(msg, "content", str(msg))})

        if self.chain:
            # Query RAG chain specifically filtered to the 'tax' category
            rag_response = self.chain.invoke(
                question=user_query,
                category="tax",
                conversation_history=conv_history,
            )
            content = rag_response.answer + TAX_DISCLAIMER
            sources = rag_response.sources
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = (
                "You are an expert financial tax educator. Explain the following tax concept clearly, "
                "addressing whether it applies to US or Indian tax law:\n"
                f"{user_query}"
            )
            resp = self.llm.invoke(
                [SystemMessage(content=prompt), HumanMessage(content=user_query)]
            )
            content = str(resp.content) + TAX_DISCLAIMER
            sources = []

        return AgentOutput(
            agent_name=self.name,
            content=content,
            sources=sources,
            data={},
            error=None,
        )
