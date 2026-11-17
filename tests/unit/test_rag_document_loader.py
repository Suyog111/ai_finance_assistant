"""
tests/unit/test_rag_document_loader.py

Unit tests for src/rag/document_loader.py
Uses temporary directories — no dependency on real article files.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.document_loader import _load_json_article, load_articles


def _write_article(directory: Path, filename: str, data: dict) -> Path:
    """Helper: write a JSON article to a tmp directory."""
    path = directory / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLoadJsonArticle:
    def test_valid_article_returns_document(self, tmp_path):
        p = _write_article(
            tmp_path,
            "test.json",
            {
                "id": "test_article",
                "title": "Test Article",
                "category": "basics",
                "source": "Test Source",
                "date": "2024-01-01",
                "tags": ["test"],
                "content": "This is a test article about investing basics." * 10,
            },
        )
        doc = _load_json_article(p)
        assert doc is not None
        assert doc.metadata["id"] == "test_article"
        assert doc.metadata["category"] == "basics"
        assert "investing" in doc.page_content

    def test_empty_content_returns_none(self, tmp_path):
        p = _write_article(
            tmp_path,
            "empty.json",
            {
                "id": "empty",
                "title": "Empty Article",
                "category": "basics",
                "content": "",
            },
        )
        doc = _load_json_article(p)
        assert doc is None

    def test_malformed_json_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json}", encoding="utf-8")
        doc = _load_json_article(p)
        assert doc is None

    def test_missing_fields_use_defaults(self, tmp_path):
        p = _write_article(
            tmp_path,
            "minimal.json",
            {
                "content": "Minimal article content that is long enough." * 5,
            },
        )
        doc = _load_json_article(p)
        assert doc is not None
        assert doc.metadata["id"] == "minimal"  # falls back to filename stem
        assert doc.metadata["source"] == "AI Finance Assistant Knowledge Base"

    def test_metadata_includes_all_fields(self, tmp_path):
        p = _write_article(
            tmp_path,
            "full.json",
            {
                "id": "full_id",
                "title": "Full Article",
                "category": "portfolio",
                "source": "My Source",
                "date": "2024-06-01",
                "tags": ["diversification", "etf"],
                "content": "Content about portfolio diversification." * 10,
            },
        )
        doc = _load_json_article(p)
        assert doc.metadata["tags"] == ["diversification", "etf"]
        assert doc.metadata["date"] == "2024-06-01"


class TestLoadArticles:
    def _build_fake_kb(self, root: Path) -> None:
        """Create a minimal fake knowledge base structure."""
        for cat in ["basics", "portfolio"]:
            cat_dir = root / cat
            cat_dir.mkdir()
            for i in range(3):
                _write_article(
                    cat_dir,
                    f"article_{i}.json",
                    {
                        "id": f"{cat}_article_{i}",
                        "title": f"{cat.title()} Article {i}",
                        "category": cat,
                        "content": f"Content for {cat} article {i}." * 20,
                    },
                )

    def test_loads_all_categories(self, tmp_path):
        self._build_fake_kb(tmp_path)
        docs = load_articles(articles_dir=tmp_path)
        assert len(docs) == 6  # 3 per category × 2 categories

    def test_category_filter_works(self, tmp_path):
        self._build_fake_kb(tmp_path)
        docs = load_articles(articles_dir=tmp_path, categories=["basics"])
        assert len(docs) == 3
        assert all(d.metadata["category"] == "basics" for d in docs)

    def test_missing_directory_returns_empty(self):
        docs = load_articles(articles_dir=Path("/nonexistent/path/xyz"))
        assert docs == []

    def test_empty_category_dir_returns_no_docs(self, tmp_path):
        (tmp_path / "basics").mkdir()  # Empty category directory
        docs = load_articles(articles_dir=tmp_path)
        assert docs == []
