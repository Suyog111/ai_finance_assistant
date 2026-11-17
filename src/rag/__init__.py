"""
src/rag/__init__.py
Exports public RAG API.
"""

from src.rag.chunker import chunk_documents
from src.rag.document_loader import load_articles, stream_articles
from src.rag.embeddings import get_embeddings
from src.rag.rag_chain import DISCLAIMER, RAGChain, RAGResponse
from src.rag.retriever import FinanceRetriever, RetrievalResult
from src.rag.vector_store import get_or_build_vector_store, load_vector_store, save_vector_store

__all__ = [
    "load_articles",
    "stream_articles",
    "chunk_documents",
    "get_embeddings",
    "get_or_build_vector_store",
    "load_vector_store",
    "save_vector_store",
    "FinanceRetriever",
    "RetrievalResult",
    "RAGChain",
    "RAGResponse",
    "DISCLAIMER",
]
