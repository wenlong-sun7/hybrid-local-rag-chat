"""Hybrid RAG engine: retrieval (BM25 + ChromaDB) and Ollama generation.

This package is deliberately free of Streamlit imports so the core logic
can be tested and reused independently of the UI.
"""
from .generation import build_prompt, stream_answer
from .retrieval import (
    RetrievalResult,
    build_bm25_index,
    get_chroma_collection,
    load_documents,
    retrieve_hybrid,
    sync_chroma,
)

__all__ = [
    "RetrievalResult",
    "build_bm25_index",
    "build_prompt",
    "get_chroma_collection",
    "load_documents",
    "retrieve_hybrid",
    "stream_answer",
    "sync_chroma",
]