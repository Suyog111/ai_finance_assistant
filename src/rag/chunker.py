"""
src/rag/chunker.py

Splits loaded Documents into smaller chunks for embedding.
Uses LangChain's RecursiveCharacterTextSplitter with
finance-aware separators to preserve sentence boundaries.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.logging_config import get_logger
from src.core.settings import get_settings

logger = get_logger(__name__)


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Return a configured RecursiveCharacterTextSplitter.

    Uses settings from config.yaml by default; override via args for testing.
    """
    settings = get_settings()
    size = chunk_size or settings.rag.chunk_size
    overlap = chunk_overlap or settings.rag.chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        # Finance-aware separators: preserve paragraphs → sentences → words
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a list of Documents into chunks, preserving all metadata.

    Each chunk inherits the parent document's metadata and gains two
    additional fields:
      - chunk_index: position of this chunk within the parent doc
      - parent_id:   the article `id` from metadata

    Args:
        documents:     List of full-article Documents.
        chunk_size:    Override chunk size (default from config).
        chunk_overlap: Override overlap (default from config).

    Returns:
        List of chunk Documents.
    """
    if not documents:
        return []

    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []

    for doc in documents:
        split_docs = splitter.split_documents([doc])
        for idx, chunk in enumerate(split_docs):
            # Enrich metadata
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["parent_id"] = doc.metadata.get("id", "unknown")
            chunk.metadata["total_chunks"] = len(split_docs)
            chunks.append(chunk)

    logger.info(
        "Chunking complete",
        input_docs=len(documents),
        output_chunks=len(chunks),
        avg_chunks_per_doc=round(len(chunks) / max(len(documents), 1), 1),
    )
    return chunks
