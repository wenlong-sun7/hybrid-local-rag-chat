# 🔍 Hybrid Local RAG Chat

A fully local Retrieval-Augmented Generation (RAG) chat application built with **Streamlit**, **ChromaDB**, **BM25**, and **Ollama**. It retrieves relevant context from your text documents using a hybrid dual-channel approach and streams answers from a local LLM.

## ✨ Features

- 💬 **Interactive chat interface** — multi-turn conversations with conversation history
- 🔤 **BM25 sparse retrieval** — keyword-based lexical matching
- 🧠 **ChromaDB dense retrieval** — semantic vector search using `all-MiniLM-L6-v2` embeddings
- 🔀 **Hybrid fusion** — combines both channels and sends the top match from each to the LLM
- ⚡ **Streaming responses** — answers stream token-by-token from Ollama
- 📄 **Transparent context** — expandable panels show exactly which documents were sent to the LLM
- 🗂️ **Auto-indexing** — new `.txt` files added to `documents/` are indexed automatically on boot

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────────────┐
│  Hybrid Retrieval (Stage 1) │
│                             │
│  ┌──────────┐  ┌─────────┐ │
│  │ BM25     │  │ ChromaDB│ │
│  │ (sparse) │  │ (dense) │ │
│  └────┬─────┘  └────┬────┘ │
│       └──────┬──────┘      │
│          Top match         │
│          per channel       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Ollama LLM (Stage 2)       │
│  qwen2.5:1.5b               │
│  + conversation history     │
└─────────────┬───────────────┘
              │
              ▼
        Streamed Answer
```

## 📋 Prerequisites

- **Python 3.9+**
- **[Ollama](https://ollama.com/)** installed and running locally
- The model pulled: `ollama pull qwen2.5:1.5b`

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone <https://github.com/wenlong-sun7/hybrid-local-rag-chat/>
cd hybrid-local-rag-chat

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running and the model is pulled
ollama serve                # in a separate terminal
ollama pull qwen2.5:1.5b

# 5. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## 📚 Adding Your Own Documents

1. Drop any `.txt` files into the `documents/` folder
2. Click **"🔄 Re-index Files"** in the sidebar (or restart the app)
3. The vector store and BM25 index rebuild automatically

## ⚙️ Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model name |
| `DOCUMENTS_DIR` | `./documents` | Folder containing `.txt` files |
| `CHROMA_PATH` | `./chroma_db` | Where the vector DB is persisted |

Example:

```bash
OLLAMA_MODEL=llama3.2:3b streamlit run app.py
```

## 🐳 Docker (Optional)

A `Dockerfile` and `docker-compose.yml` are included for a fully containerized setup:

```bash
docker compose up --build
```

This starts both the Streamlit app and an Ollama service. The first run will pull the model (may take a few minutes).

## 🗂️ Project Structure

```
hybrid-local-rag-chat/
├── app.py               # Streamlit UI layer (chat interface)
├── rag/                 # Core engine (no Streamlit dependency)
│   ├── __init__.py      # Public API exports
│   ├── config.py        # Environment-based configuration
│   ├── retrieval.py     # Document loading, indexing, hybrid search
│   └── generation.py    # Prompt building & Ollama streaming
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore rules
├── Dockerfile           # App container
├── docker-compose.yml   # App + Ollama orchestration
├── documents/           # Your .txt source documents
└── chroma_db/           # (auto-generated) Vector DB — not committed
```

## 🧠 How It Works

1. **Indexing** — On boot, all `.txt` files in `documents/` are loaded, embedded with `all-MiniLM-L6-v2`, and stored in a persistent ChromaDB collection. A BM25 inverted index is also built in memory.

2. **Retrieval** — For each user question, two channels run in parallel:
   - **BM25** scores documents by keyword overlap
   - **ChromaDB** scores documents by semantic similarity
   
   The top match from each channel (deduplicated) is selected as context.

3. **Generation** — The selected documents + conversation history are injected into a prompt sent to Ollama (`qwen2.5:1.5b`), which streams a grounded answer.

## 📄 License

MIT — see [LICENSE](LICENSE).

## ⚠️ Notes

- The app is designed for **local use** — it requires a running Ollama instance, so it won't work on a free public PaaS out of the box.
- The `chroma_db/` directory is auto-generated and excluded from version control.
- For better answer quality, consider a larger model (e.g., `qwen2.5:3b`, `llama3.2:3b`) — trade-off is slower inference.
