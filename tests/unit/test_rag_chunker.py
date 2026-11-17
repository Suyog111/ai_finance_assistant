"""
tests/unit/test_rag_chunker.py

Unit tests for src/rag/chunker.py
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.rag.chunker import chunk_documents, get_text_splitter


class TestGetTextSplitter:
    def test_returns_splitter_with_config_defaults(self):
        splitter = get_text_splitter()
        assert splitter._chunk_size == 512
        assert splitter._chunk_overlap == 64

    def test_respects_override_params(self):
        splitter = get_text_splitter(chunk_size=256, chunk_overlap=32)
        assert splitter._chunk_size == 256
        assert splitter._chunk_overlap == 32


class TestChunkDocuments:
    def _make_doc(self, content: str, category: str = "basics", doc_id: str = "doc1") -> Document:
        return Document(
            page_content=content,
            metadata={"id": doc_id, "category": category, "title": "Test Article"},
        )

    def test_empty_input_returns_empty(self):
        assert chunk_documents([]) == []

    def test_short_doc_returns_single_chunk(self):
        doc = self._make_doc("This is a short article about stocks.")
        chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1

    def test_long_doc_produces_multiple_chunks(self):
        long_content = "Investing is the act of allocating resources. " * 60  # ~2700 chars
        doc = self._make_doc(long_content)
        chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=64)
        assert len(chunks) > 1

    def test_chunks_inherit_parent_metadata(self):
        doc = self._make_doc("Content. " * 100, category="tax", doc_id="tax_001")
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        for chunk in chunks:
            assert chunk.metadata["category"] == "tax"
            assert chunk.metadata["parent_id"] == "tax_001"

    def test_chunks_have_index_metadata(self):
        doc = self._make_doc("Word " * 300)
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_have_total_chunks_metadata(self):
        doc = self._make_doc("Word " * 300)
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        total = len(chunks)
        assert all(c.metadata["total_chunks"] == total for c in chunks)

    def test_multiple_docs_all_chunked(self):
        docs = [self._make_doc(f"Content {i}. " * 50, doc_id=f"doc{i}") for i in range(3)]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        parent_ids = {c.metadata["parent_id"] for c in chunks}
        assert parent_ids == {"doc0", "doc1", "doc2"}
