"""Retrieval pipeline: document loading, indexing, and hybrid search.

Two complementary channels are used:
- **BM25** (sparse): keyword/lexical matching via rank_bm25.
- **ChromaDB** (dense): semantic vector search with Sentence Transformers.

The top result from each channel is returned so the UI can show exactly
which documents were selected (and sent to the LLM).
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from . import config


# ---------------------------------------------------------------------------
# Documents & indexing
# ---------------------------------------------------------------------------
@dataclass
class DocumentIndex:
    """In-memory representation of the indexed document collection."""

    docs: List[str] = field(default_factory=list)
    file_ids: List[str] = field(default_factory=list)
    bm25: Optional[BM25Okapi] = None

    @property
    def is_empty(self) -> bool:
        return not self.docs

    def lookup(self, file_id: str) -> Optional[str]:
        """Return the text of a document by its file id."""
        try:
            idx = self.file_ids.index(file_id)
            return self.docs[idx]
        except ValueError:
            return None


def get_chroma_collection():
    """Return (and lazily create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL_NAME
    )
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def load_documents(folder_path: str) -> DocumentIndex:
    """Load all ``.txt`` files from ``folder_path`` into a DocumentIndex.

    Also syncs the ChromaDB vector store with any new documents.
    """
    os.makedirs(folder_path, exist_ok=True)

    docs: List[str] = []
    file_ids: List[str] = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())
                file_ids.append(filename)

    index = DocumentIndex(docs=docs, file_ids=file_ids)
    if index.is_empty:
        return index

    sync_chroma(file_ids, docs)
    index.bm25 = build_bm25_index(docs)

    return index


def build_bm25_index(docs: List[str]) -> BM25Okapi:
    """Build a BM25 inverted index from a list of document texts."""
    tokenized_corpus = [doc.lower().split() for doc in docs]
    return BM25Okapi(tokenized_corpus)


def sync_chroma(file_ids: List[str], docs: List[str]) -> None:
    """Add any documents not yet present in the ChromaDB collection."""
    collection = get_chroma_collection()
    existing_ids = set(collection.get()["ids"])

    new_docs = [doc for fid, doc in zip(file_ids, docs) if fid not in existing_ids]
    new_ids = [fid for fid in file_ids if fid not in existing_ids]

    if new_ids:
        collection.add(documents=new_docs, ids=new_ids)


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """Result of a hybrid retrieval query."""

    matched_contexts: Dict[str, str] = field(default_factory=dict)
    bm25_id: Optional[str] = None
    emb_id: Optional[str] = None


def _rrf_add(scores: Dict[str, float], file_id: str, rank: int, k: int) -> None:
    """Add one Reciprocal Rank Fusion contribution."""
    scores[file_id] = scores.get(file_id, 0) + (1 / (k + rank + 1))


def _select_contexts(index: DocumentIndex, *, bm25_id, emb_id) -> Dict[str, str]:
    """Map the chosen channel winners to their full document text."""
    selected_ids: List[str] = []
    if bm25_id:
        selected_ids.append(bm25_id)
    if emb_id and emb_id != bm25_id:
        selected_ids.append(emb_id)

    return {fid: index.lookup(fid) for fid in selected_ids if index.lookup(fid) is not None}


def retrieve_hybrid(index: DocumentIndex, query: str) -> RetrievalResult:
    """Run BM25 and ChromaDB retrieval in parallel and fuse via RRF.

    Only the **top match from each channel** (deduplicated) is returned,
    so the context sent to the LLM always matches what the UI displays.
    """
    result = RetrievalResult()

    if index.is_empty or index.bm25 is None:
        return result

    fused_scores: Dict[str, float] = {}

    # --- Channel A: BM25 (sparse keyword search) ---
    query_tokens = query.lower().split()
    bm25_scores = index.bm25.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][: config.BM25_TOP_K]

    if bm25_scores[top_bm25_indices[0]] > 0:
        result.bm25_id = index.file_ids[top_bm25_indices[0]]
        for rank, idx in enumerate(top_bm25_indices):
            if bm25_scores[idx] > 0:
                fid = index.file_ids[idx]
                _rrf_add(fused_scores, fid, rank, config.HYBRID_RRF_K)

    # --- Channel B: ChromaDB (dense semantic search) ---
    collection = get_chroma_collection()
    chroma_results = collection.query(query_texts=[query], n_results=config.EMBEDDING_TOP_K)

    if chroma_results["ids"] and chroma_results["ids"][0]:
        result.emb_id = chroma_results["ids"][0][0]
        for rank, fid in enumerate(chroma_results["ids"][0]):
            _rrf_add(fused_scores, fid, rank, config.HYBRID_RRF_K)

    result.matched_contexts = _select_contexts(
        index, bm25_id=result.bm25_id, emb_id=result.emb_id
    )

    return result