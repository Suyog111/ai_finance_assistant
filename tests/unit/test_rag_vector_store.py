"""
tests/unit/test_rag_vector_store.py

Unit tests for src/rag/vector_store.py
Uses mock embeddings and mock vector stores.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.vector_store import (
    build_vector_store,
    get_or_build_vector_store,
    load_vector_store,
    save_vector_store,
)


@pytest.fixture
def sample_chunks():
    return [
        Document(
            page_content="Test chunk about index funds and diversification.",
            metadata={"id": "doc1", "category": "basics"},
        )
    ]


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    emb.embed_query.return_value = [0.1, 0.2, 0.3]
    return emb


class TestVectorStore:
    def test_build_faiss_store(self, sample_chunks, mock_embeddings):
        mock_faiss = MagicMock()
        with patch(
            "langchain_community.vectorstores.FAISS.from_documents", return_value=mock_faiss
        ) as mock_builder:
            store = build_vector_store(
                sample_chunks, embeddings=mock_embeddings, store_type="faiss"
            )
            assert store is mock_faiss
            mock_builder.assert_called_once_with(sample_chunks, mock_embeddings)

    def test_save_faiss_store(self, tmp_path):
        mock_store = MagicMock()
        save_path = tmp_path / "faiss_test"

        with patch("src.rag.vector_store.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.vector_store.type = "faiss"
            mock_settings.vector_store.index_path = save_path
            mock_get_settings.return_value = mock_settings

            save_vector_store(mock_store, store_type="faiss")
            mock_store.save_local.assert_called_once_with(str(save_path))

    def test_load_faiss_raises_if_not_found(self, mock_embeddings, tmp_path):
        non_existent = tmp_path / "does_not_exist"
        with patch("src.rag.vector_store.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.vector_store.type = "faiss"
            mock_settings.vector_store.index_path = non_existent
            mock_get_settings.return_value = mock_settings

            with pytest.raises(FileNotFoundError):
                load_vector_store(embeddings=mock_embeddings, store_type="faiss")

    def test_load_faiss_success(self, mock_embeddings, tmp_path):
        index_dir = tmp_path / "faiss_idx"
        index_dir.mkdir()
        mock_store = MagicMock()

        with patch("src.rag.vector_store.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.vector_store.type = "faiss"
            mock_settings.vector_store.index_path = index_dir
            mock_get_settings.return_value = mock_settings

            with patch(
                "langchain_community.vectorstores.FAISS.load_local", return_value=mock_store
            ) as mock_loader:
                store = load_vector_store(embeddings=mock_embeddings, store_type="faiss")
                assert store is mock_store
                mock_loader.assert_called_once()

    def test_get_or_build_loads_existing(self, mock_embeddings, tmp_path):
        index_dir = tmp_path / "existing_index"
        index_dir.mkdir()
        mock_store = MagicMock()

        with patch("src.rag.vector_store.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.vector_store.type = "faiss"
            mock_settings.vector_store.index_path = index_dir
            mock_get_settings.return_value = mock_settings

            with patch(
                "langchain_community.vectorstores.FAISS.load_local", return_value=mock_store
            ):
                store = get_or_build_vector_store(embeddings=mock_embeddings, store_type="faiss")
                assert store is mock_store

    def test_get_or_build_raises_when_no_index_and_no_chunks(self, mock_embeddings, tmp_path):
        missing_dir = tmp_path / "not_there"

        with patch("src.rag.vector_store.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.vector_store.type = "faiss"
            mock_settings.vector_store.index_path = missing_dir
            mock_get_settings.return_value = mock_settings

            with pytest.raises(ValueError, match="No vector store found"):
                get_or_build_vector_store(
                    chunks=None, embeddings=mock_embeddings, store_type="faiss"
                )
