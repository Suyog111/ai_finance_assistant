"""
src/rag/document_loader.py

Loads financial education articles from the data/articles directory.
Each JSON article becomes a LangChain Document with rich metadata.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Path to articles relative to project root
ARTICLES_DIR = Path(__file__).resolve().parents[2] / "src" / "data" / "articles"

VALID_CATEGORIES = {"basics", "portfolio", "tax", "market", "goals", "news"}


def _load_json_article(path: Path) -> Document | None:
    """
    Parse a single JSON article file into a LangChain Document.

    Returns None and logs a warning if the file is malformed.
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        content = data.get("content", "").strip()
        if not content:
            logger.warning("Article has empty content, skipping", path=str(path))
            return None

        metadata = {
            "id": data.get("id", path.stem),
            "title": data.get("title", path.stem.replace("_", " ").title()),
            "category": data.get("category", path.parent.name),
            "source": data.get("source", "AI Finance Assistant Knowledge Base"),
            "date": data.get("date", ""),
            "tags": data.get("tags", []),
            "file_path": str(path),
        }

        return Document(page_content=content, metadata=metadata)

    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in article, skipping", path=str(path), error=str(exc))
        return None
    except Exception as exc:
        logger.error("Unexpected error loading article", path=str(path), error=str(exc))
        return None


def load_articles(
    articles_dir: Path | None = None,
    categories: list[str] | None = None,
) -> list[Document]:
    """
    Load all JSON articles from the knowledge base directory.

    Args:
        articles_dir: Root directory containing category subdirectories.
                      Defaults to src/data/articles.
        categories:   Filter to specific categories. None loads all.

    Returns:
        List of LangChain Documents, one per article.
    """
    base = Path(articles_dir) if articles_dir else ARTICLES_DIR

    if not base.exists():
        logger.error("Articles directory not found", path=str(base))
        return []

    target_categories = set(categories) if categories else VALID_CATEGORIES
    documents: list[Document] = []

    for category_dir in sorted(base.iterdir()):
        if not category_dir.is_dir():
            continue
        if category_dir.name not in target_categories:
            continue

        json_files = sorted(category_dir.glob("*.json"))
        if not json_files:
            logger.warning("No JSON files found in category", category=category_dir.name)
            continue

        for json_path in json_files:
            doc = _load_json_article(json_path)
            if doc:
                documents.append(doc)

        logger.info(
            "Loaded category",
            category=category_dir.name,
            files=len(json_files),
            loaded=sum(1 for d in documents if d.metadata["category"] == category_dir.name),
        )

    logger.info("Total articles loaded", count=len(documents))
    return documents


def stream_articles(
    articles_dir: Path | None = None,
    categories: list[str] | None = None,
) -> Iterator[Document]:
    """Generator version of load_articles — memory-efficient for large corpora."""
    base = Path(articles_dir) if articles_dir else ARTICLES_DIR
    target_categories = set(categories) if categories else VALID_CATEGORIES

    if not base.exists():
        return

    for category_dir in sorted(base.iterdir()):
        if category_dir.is_dir() and category_dir.name in target_categories:
            for json_path in sorted(category_dir.glob("*.json")):
                doc = _load_json_article(json_path)
                if doc:
                    yield doc
