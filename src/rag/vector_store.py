"""
src/rag/vector_store.py

Builds, persists, and loads the FAISS vector index.
Supports optional ChromaDB as an alternative persistent store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.core.logging_config import get_logger
from src.core.settings import get_settings
from src.rag.embeddings import get_embeddings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------


def _build_faiss(chunks: list[Document], embeddings: Embeddings) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    logger.info("Building FAISS index", chunks=len(chunks))
    store = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built successfully")
    return store


def _save_faiss(store: VectorStore, index_path: Path) -> None:
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))  # type: ignore[attr-defined]
    logger.info("FAISS index saved", path=str(index_path))


def _load_faiss(index_path: Path, embeddings: Embeddings) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")

    store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded", path=str(index_path))
    return store


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


def _build_chroma(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_dir: Path,
    collection_name: str,
) -> VectorStore:
    from langchain_community.vectorstores import Chroma

    persist_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Building ChromaDB collection", chunks=len(chunks), collection=collection_name)
    store = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    logger.info("ChromaDB collection built and persisted")
    return store


def _load_chroma(
    embeddings: Embeddings,
    persist_dir: Path,
    collection_name: str,
) -> VectorStore:
    from langchain_community.vectorstores import Chroma

    if not persist_dir.exists():
        raise FileNotFoundError(f"ChromaDB persist directory not found at: {persist_dir}")

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    logger.info("ChromaDB collection loaded", collection=collection_name)
    return store


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_vector_store(
    chunks: list[Document],
    embeddings: Embeddings | None = None,
    store_type: Literal["faiss", "chroma"] | None = None,
) -> VectorStore:
    """
    Build a vector store from document chunks.

    Args:
        chunks:     Chunked Documents to embed and index.
        embeddings: Embeddings instance. Defaults to get_embeddings().
        store_type: "faiss" or "chroma". Defaults to config setting.

    Returns:
        A LangChain VectorStore instance (not yet persisted for FAISS).
    """
    settings = get_settings()
    emb = embeddings or get_embeddings()
    stype = store_type or settings.vector_store.type

    if stype == "faiss":
        return _build_faiss(chunks, emb)
    elif stype == "chroma":
        return _build_chroma(
            chunks,
            emb,
            settings.vector_store.chroma_persist_dir,
            settings.vector_store.collection_name,
        )
    else:
        raise ValueError(f"Unknown vector store type: '{stype}'")


def save_vector_store(
    store: VectorStore, store_type: Literal["faiss", "chroma"] | None = None
) -> None:
    """Persist the vector store to disk."""
    settings = get_settings()
    stype = store_type or settings.vector_store.type

    if stype == "faiss":
        _save_faiss(store, settings.vector_store.index_path)
    elif stype == "chroma":
        # ChromaDB auto-persists on build; this is a no-op
        logger.info("ChromaDB auto-persists on creation — no manual save needed")
    else:
        raise ValueError(f"Unknown store type: '{stype}'")


def load_vector_store(
    embeddings: Embeddings | None = None,
    store_type: Literal["faiss", "chroma"] | None = None,
) -> VectorStore:
    """
    Load a previously built vector store from disk.

    Raises FileNotFoundError if the index doesn't exist — call build first.
    """
    settings = get_settings()
    emb = embeddings or get_embeddings()
    stype = store_type or settings.vector_store.type

    if stype == "faiss":
        return _load_faiss(settings.vector_store.index_path, emb)
    elif stype == "chroma":
        return _load_chroma(
            emb,
            settings.vector_store.chroma_persist_dir,
            settings.vector_store.collection_name,
        )
    else:
        raise ValueError(f"Unknown store type: '{stype}'")


def get_or_build_vector_store(
    chunks: list[Document] | None = None,
    embeddings: Embeddings | None = None,
    store_type: Literal["faiss", "chroma"] | None = None,
    force_rebuild: bool = False,
) -> VectorStore:
    """
    Load from disk if available; build and save if not (or if force_rebuild=True).

    This is the main entry point used by agents and the app.
    If chunks is None and no saved index exists, raises FileNotFoundError.
    """
    settings = get_settings()
    stype = store_type or settings.vector_store.type

    # Check whether a saved index exists
    index_exists = (
        settings.vector_store.index_path.exists()
        if stype == "faiss"
        else settings.vector_store.chroma_persist_dir.exists()
    )

    if index_exists and not force_rebuild:
        logger.info("Loading existing vector store from disk", type=stype)
        return load_vector_store(embeddings, stype)

    if not chunks:
        raise ValueError(
            "No vector store found on disk and no chunks provided to build one. "
            "Run the index builder script first: python -m src.rag.build_index"
        )

    logger.info("Building new vector store", type=stype, chunks=len(chunks))
    store = build_vector_store(chunks, embeddings, stype)
    save_vector_store(store, stype)
    return store
