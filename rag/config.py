"""Application configuration loaded from environment variables.

All settings can be overridden with environment variables, which makes the
app easy to configure both locally and in Docker.
"""
import os

# Suppress Hugging Face transformers internal docstring/warning logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# --- Paths ---
DATA_DIR = os.getenv("DOCUMENTS_DIR", "./documents")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# --- Ollama ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# --- Retrieval tuning ---
# How many candidates each channel contributes before fusion.
BM25_TOP_K = 5
EMBEDDING_TOP_K = 5
HYBRID_RRF_K = 60

# --- Embedding model (Sentence Transformers via ChromaDB) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "txt_documents"

# --- LLM generation ---
LLM_MAX_TOKENS = 120
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT_SECONDS = 15
# How many past messages to include in the prompt for conversational context.
HISTORY_TURNS = 6