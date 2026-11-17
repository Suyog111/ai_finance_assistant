"""
src/rag/rag_chain.py

Full RAG chain: query → retrieve → prompt → LLM → answer + citations.
Includes regulatory disclaimer injection for all financial responses.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.core.logging_config import get_logger
from src.rag.retriever import FinanceRetriever, RetrievalResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regulatory disclaimer — injected into every financial response
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Disclaimer:** This information is for educational purposes only and does not "
    "constitute financial, investment, tax, or legal advice. Always consult a qualified "
    "financial advisor before making investment decisions. Past performance does not "
    "guarantee future results."
)

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = """You are an expert financial educator helping beginners learn about \
personal finance and investing. Your goal is to educate, not to advise.

GUIDELINES:
- Explain concepts clearly using simple language and relatable examples
- Always ground your answer in the provided knowledge base excerpts
- Cite sources by mentioning article titles naturally in your response
- If the knowledge base does not contain enough information, say so honestly
- Never make specific investment recommendations (e.g., "you should buy X")
- Keep responses structured and easy to scan (use bullet points or numbered lists when helpful)
- Adapt your explanation to a beginner-friendly level unless context suggests otherwise

KNOWLEDGE BASE EXCERPTS:
{context}
"""

USER_TEMPLATE = "{question}"


@dataclass
class RAGResponse:
    """Complete response from the RAG chain."""

    answer: str  # LLM-generated answer
    sources: list[dict]  # [{title, category, source, id}]
    context_used: str  # Raw context passed to LLM
    retrieval: RetrievalResult

    @property
    def answer_with_disclaimer(self) -> str:
        return self.answer + DISCLAIMER

    @property
    def formatted_sources(self) -> str:
        if not self.sources:
            return ""
        lines = ["**Sources:**"]
        for i, s in enumerate(self.sources, 1):
            lines.append(f"{i}. *{s['title']}* — {s['category'].title()} | {s['source']}")
        return "\n".join(lines)


class RAGChain:
    """
    Retrieval-Augmented Generation chain for financial Q&A.

    Usage:
        chain = RAGChain(retriever, llm)
        response = chain.invoke("What is a Roth IRA?", category="tax")
        print(response.answer_with_disclaimer)
        print(response.formatted_sources)
    """

    def __init__(
        self,
        retriever: FinanceRetriever,
        llm: BaseChatModel,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_TEMPLATE),
                ("human", USER_TEMPLATE),
            ]
        )
        self._parser = StrOutputParser()

    def invoke(
        self,
        question: str,
        category: str | None = None,
        top_k: int | None = None,
        conversation_history: list[dict] | None = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a user question.

        Args:
            question:             User's question.
            category:             Optional knowledge base category filter.
            top_k:                Number of chunks to retrieve.
            conversation_history: Prior messages for multi-turn context
                                  [{role: "user"|"assistant", content: "..."}]

        Returns:
            RAGResponse with answer, sources, and retrieval metadata.
        """
        # 1. Retrieve relevant chunks
        retrieval = self.retriever.retrieve(question, category=category, top_k=top_k)

        if not retrieval.documents:
            logger.warning("No documents retrieved for query", query=question)
            answer = (
                "I couldn't find specific information about that in my knowledge base. "
                "Could you try rephrasing, or ask about a related financial topic?"
            )
            return RAGResponse(
                answer=answer,
                sources=[],
                context_used="",
                retrieval=retrieval,
            )

        context = retrieval.context_text()

        # 2. Build messages for LLM
        system_content = SYSTEM_TEMPLATE.format(context=context)
        messages = [SystemMessage(content=system_content)]

        # Inject conversation history for multi-turn context
        if conversation_history:
            from langchain_core.messages import AIMessage

            for turn in conversation_history[-6:]:  # Last 3 exchanges (6 messages)
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=question))

        # 3. Generate answer
        logger.info("Invoking LLM", question=question[:60], context_chunks=len(retrieval.documents))
        try:
            response = self.llm.invoke(messages)
            answer = self._parser.invoke(response)
        except Exception as exc:
            logger.error("LLM invocation failed", error=str(exc))
            answer = (
                "I encountered an issue generating a response. " "Please try again in a moment."
            )

        return RAGResponse(
            answer=answer,
            sources=retrieval.sources,
            context_used=context,
            retrieval=retrieval,
        )
