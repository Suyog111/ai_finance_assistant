"""
src/rag/build_index.py

CLI script to build (or rebuild) the FAISS vector index from scratch.
Run as: python -m src.rag.build_index [--force]

This script:
  1. Loads all articles from src/data/articles/
  2. Chunks them
  3. Embeds with OpenAI text-embedding-3-small
  4. Saves FAISS index to data/faiss_index/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logging_config import get_logger, setup_logging
from src.core.settings import get_settings
from src.rag.chunker import chunk_documents
from src.rag.document_loader import load_articles
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import build_vector_store, save_vector_store

setup_logging(level="INFO", fmt="text")
logger = get_logger(__name__)


def build_index(force: bool = False) -> None:
    settings = get_settings()
    index_path = settings.vector_store.index_path

    if index_path.exists() and not force:
        logger.info(
            "Index already exists. Use --force to rebuild.",
            path=str(index_path),
        )
        return

    start = time.perf_counter()

    # Step 1: Load articles
    logger.info("Loading articles...")
    documents = load_articles()
    if not documents:
        logger.error("No articles found! Add JSON files to src/data/articles/")
        sys.exit(1)
    logger.info(f"Loaded {len(documents)} articles")

    # Step 2: Chunk
    logger.info("Chunking documents...")
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # Step 3: Embed + build index
    logger.info("Building vector index (this may take a moment)...")
    embeddings = get_embeddings()
    store = build_vector_store(chunks, embeddings)

    # Step 4: Save
    save_vector_store(store)

    elapsed = time.perf_counter() - start
    logger.info(
        "Index build complete",
        articles=len(documents),
        chunks=len(chunks),
        elapsed_seconds=round(elapsed, 1),
        index_path=str(index_path),
    )

    # Print summary
    print("\n" + "=" * 55)
    print("  ✅ FAISS Index Built Successfully")
    print("=" * 55)
    print(f"  Articles loaded : {len(documents)}")
    print(f"  Chunks created  : {len(chunks)}")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print(f"  Index saved to  : {index_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector index")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if index already exists",
    )
    args = parser.parse_args()
    build_index(force=args.force)
