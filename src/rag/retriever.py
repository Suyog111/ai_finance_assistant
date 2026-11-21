"""
src/rag/retriever.py

Category-filtered semantic retriever with MMR (Maximal Marginal Relevance)
for diversity. Returns documents with full source attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.core.logging_config import get_logger
from src.core.settings import get_settings

logger = get_logger(__name__)

VALID_CATEGORIES = {"basics", "portfolio", "tax", "market", "goals", "news"}


@dataclass
class RetrievalResult:
    """Structured retrieval output returned to agents."""

    documents: list[Document]
    query: str
    category_filter: Optional[str]
    total_found: int
    sources: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Build the deduplicated sources list from documents."""
        seen: set[str] = set()
        for doc in self.documents:
            article_id = doc.metadata.get("id", "")
            if article_id not in seen:
                seen.add(article_id)
                self.sources.append(
                    {
                        "title": doc.metadata.get("title", "Unknown"),
                        "category": doc.metadata.get("category", ""),
                        "source": doc.metadata.get("source", "Knowledge Base"),
                        "id": article_id,
                    }
                )

    def context_text(self) -> str:
        """Return concatenated chunk text for LLM context injection."""
        parts = []
        for i, doc in enumerate(self.documents, 1):
            title = doc.metadata.get("title", "Article")
            parts.append(f"[Source {i}: {title}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


class FinanceRetriever:
    """
    Semantic retriever over the financial knowledge base.

    Supports:
    - Category filtering (basics, portfolio, tax, market, goals, news)
    - MMR search for result diversity
    - Fallback to similarity search if MMR fails
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.store = vector_store
        self.settings = get_settings()

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int | None = None,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
    ) -> RetrievalResult:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query:      User's question or topic.
            category:   Optional category filter ("basics", "tax", etc.).
            top_k:      Number of chunks to return. Defaults to config rag.top_k.
            use_mmr:    Use Maximal Marginal Relevance for diversity (default True).
            mmr_lambda: MMR diversity parameter (0=max diversity, 1=max relevance).

        Returns:
            RetrievalResult with documents and source metadata.
        """
        k = top_k or self.settings.rag.top_k

        # Build metadata filter for category
        filter_dict: dict[str, Any] | None = None
        if category and category in VALID_CATEGORIES:
            filter_dict = {"category": category}
        elif category:
            logger.warning("Unknown category filter, ignoring", category=category)

        try:
            if use_mmr:
                docs = self._mmr_search(query, k, filter_dict, mmr_lambda)
            else:
                docs = self._similarity_search(query, k, filter_dict)

        except Exception as exc:
            logger.error("Retrieval failed", query=query, error=str(exc))
            # Return empty result rather than raising
            return RetrievalResult(
                documents=[],
                query=query,
                category_filter=category,
                total_found=0,
            )

        logger.info(
            "Retrieved documents",
            query=query[:60],
            category=category,
            found=len(docs),
        )

        return RetrievalResult(
            documents=docs,
            query=query,
            category_filter=category,
            total_found=len(docs),
        )

    def _mmr_search(
        self,
        query: str,
        k: int,
        filter_dict: dict | None,
        lambda_mult: float,
    ) -> list[Document]:
        """Max Marginal Relevance search — balances relevance with diversity."""
        fetch_k = max(k * 4, 20)  # Fetch more candidates for MMR to select from
        try:
            return self.store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_dict,
            )
        except (TypeError, NotImplementedError):
            # Some store implementations don't support filter in MMR
            logger.debug("MMR with filter not supported, falling back to similarity")
            return self._similarity_search(query, k, filter_dict)

    def _similarity_search(
        self,
        query: str,
        k: int,
        filter_dict: dict | None,
    ) -> list[Document]:
        """Standard cosine similarity search."""
        try:
            return self.store.similarity_search(query, k=k, filter=filter_dict)
        except TypeError:
            # filter kwarg not supported by this store implementation
            docs = self.store.similarity_search(query, k=k * 2)
            if filter_dict:
                cat = filter_dict.get("category")
                docs = [d for d in docs if d.metadata.get("category") == cat]
            return docs[:k]
