"""
tests/unit/test_rag_retriever.py

Unit tests for src/rag/retriever.py
Uses a mock VectorStore — no real embeddings or index needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.rag.retriever import FinanceRetriever, RetrievalResult


def _make_doc(content: str, category: str, article_id: str, title: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "id": article_id,
            "title": title,
            "category": category,
            "source": "Knowledge Base",
            "chunk_index": 0,
        },
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.similarity_search.return_value = [
        _make_doc("Stocks are equity securities.", "basics", "stocks_01", "Stocks Explained"),
        _make_doc("Bonds pay fixed interest.", "basics", "bonds_01", "Bonds Explained"),
    ]
    store.max_marginal_relevance_search.return_value = [
        _make_doc("Stocks are equity securities.", "basics", "stocks_01", "Stocks Explained"),
    ]
    return store


@pytest.fixture
def retriever(mock_store):
    return FinanceRetriever(vector_store=mock_store)


class TestRetrievalResult:
    def test_sources_built_from_documents(self):
        docs = [
            _make_doc("Content A", "basics", "art_1", "Article 1"),
            _make_doc("Content B", "basics", "art_1", "Article 1"),  # Same article, different chunk
            _make_doc("Content C", "portfolio", "art_2", "Article 2"),
        ]
        result = RetrievalResult(documents=docs, query="test", category_filter=None, total_found=3)
        assert len(result.sources) == 2  # Deduplicated
        ids = [s["id"] for s in result.sources]
        assert "art_1" in ids and "art_2" in ids

    def test_context_text_includes_all_docs(self):
        docs = [
            _make_doc("Stocks are equity.", "basics", "s1", "Stocks"),
            _make_doc("Bonds are debt.", "basics", "b1", "Bonds"),
        ]
        result = RetrievalResult(documents=docs, query="q", category_filter=None, total_found=2)
        ctx = result.context_text()
        assert "Stocks are equity." in ctx
        assert "Bonds are debt." in ctx
        assert "[Source 1:" in ctx
        assert "[Source 2:" in ctx


class TestFinanceRetriever:
    def test_retrieve_returns_result(self, retriever):
        result = retriever.retrieve("What are stocks?")
        assert isinstance(result, RetrievalResult)
        assert result.total_found > 0

    def test_retrieve_uses_mmr_by_default(self, retriever, mock_store):
        retriever.retrieve("stocks question")
        mock_store.max_marginal_relevance_search.assert_called_once()

    def test_retrieve_similarity_when_mmr_false(self, retriever, mock_store):
        retriever.retrieve("stocks question", use_mmr=False)
        mock_store.similarity_search.assert_called_once()

    def test_retrieve_passes_category_filter(self, retriever, mock_store):
        mock_store.max_marginal_relevance_search.return_value = [
            _make_doc("Tax content", "tax", "tax_1", "Tax Guide"),
        ]
        result = retriever.retrieve("tax question", category="tax")
        assert result.category_filter == "tax"

    def test_invalid_category_ignored(self, retriever, mock_store):
        """Invalid category should log warning and not crash."""
        result = retriever.retrieve("question", category="invalid_category_xyz")
        assert result is not None

    def test_retrieval_error_returns_empty_result(self, retriever, mock_store):
        mock_store.max_marginal_relevance_search.side_effect = Exception("DB error")
        mock_store.similarity_search.side_effect = Exception("DB error")
        result = retriever.retrieve("failing query")
        assert result.total_found == 0
        assert result.documents == []

    def test_sources_populated_on_success(self, retriever):
        result = retriever.retrieve("stocks question")
        assert len(result.sources) > 0
        assert "title" in result.sources[0]
