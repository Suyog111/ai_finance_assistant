"""
tests/unit/test_rag_chain.py

Unit tests for src/rag/rag_chain.py
Mocks both retriever and LLM — no API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.rag.rag_chain import DISCLAIMER, RAGChain, RAGResponse
from src.rag.retriever import RetrievalResult


def _make_retrieval(has_docs: bool = True) -> RetrievalResult:
    docs = []
    if has_docs:
        docs = [
            Document(
                page_content="Stocks represent ownership in a company.",
                metadata={"id": "s1", "title": "Stocks 101", "category": "basics", "source": "KB"},
            )
        ]
    return RetrievalResult(
        documents=docs,
        query="What are stocks?",
        category_filter=None,
        total_found=len(docs),
    )


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.retrieve.return_value = _make_retrieval(has_docs=True)
    return r


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Stocks are shares of company ownership.")
    return llm


@pytest.fixture
def chain(mock_retriever, mock_llm):
    return RAGChain(retriever=mock_retriever, llm=mock_llm)


class TestRAGChain:
    def test_invoke_returns_rag_response(self, chain):
        resp = chain.invoke("What are stocks?")
        assert isinstance(resp, RAGResponse)

    def test_answer_contains_llm_content(self, chain, mock_llm):
        resp = chain.invoke("What are stocks?")
        assert "Stocks are shares" in resp.answer

    def test_sources_populated_from_retrieval(self, chain):
        resp = chain.invoke("What are stocks?")
        assert len(resp.sources) > 0
        assert resp.sources[0]["title"] == "Stocks 101"

    def test_answer_with_disclaimer_appended(self, chain):
        resp = chain.invoke("What are stocks?")
        assert DISCLAIMER in resp.answer_with_disclaimer

    def test_no_docs_returns_fallback_message(self, chain, mock_retriever):
        mock_retriever.retrieve.return_value = _make_retrieval(has_docs=False)
        resp = chain.invoke("random unknown topic")
        assert "couldn't find" in resp.answer.lower()
        assert resp.sources == []

    def test_llm_error_returns_error_message(self, chain, mock_llm):
        mock_llm.invoke.side_effect = Exception("API timeout")
        resp = chain.invoke("What are bonds?")
        assert "encountered an issue" in resp.answer.lower()

    def test_category_passed_to_retriever(self, chain, mock_retriever):
        chain.invoke("What are Roth IRAs?", category="tax")
        mock_retriever.retrieve.assert_called_with(
            "What are Roth IRAs?", category="tax", top_k=None
        )

    def test_formatted_sources_output(self, chain):
        resp = chain.invoke("What are stocks?")
        formatted = resp.formatted_sources
        assert "Stocks 101" in formatted
        assert "**Sources:**" in formatted

    def test_formatted_sources_empty_when_no_docs(self, chain, mock_retriever):
        mock_retriever.retrieve.return_value = _make_retrieval(has_docs=False)
        resp = chain.invoke("unknown")
        assert resp.formatted_sources == ""

    def test_conversation_history_included_in_messages(self, chain, mock_llm):
        history = [
            {"role": "user", "content": "What is investing?"},
            {"role": "assistant", "content": "Investing is allocating resources."},
        ]
        chain.invoke("Tell me more", conversation_history=history)
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        # Should have system + history (user+ai) + new user = 4 messages
        assert len(messages) >= 3
