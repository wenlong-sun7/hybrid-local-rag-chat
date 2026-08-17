"""Streamlit chat UI for the hybrid RAG engine.

This module only handles the UI. Retrieval and generation logic live in the
``rag`` package (outside Streamlit) so they can be reused and tested
independently.
"""
import streamlit as st

from rag import config, load_documents, retrieve_hybrid, stream_answer

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Hybrid Local RAG Chat", page_icon="🔍")

SAMPLE_QUERIES = [
    "What are large language models?",
    "How do AI agents work?",
    "What is multimodal AI?",
    "What are the benefits of AI in healthcare?",
    "What is edge AI?",
]

NO_MATCH_RESPONSE = (
    "⚠️ No relevant documents found for this query. "
    "Please try rephrasing or ask about a different topic."
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_document_index():
    """Load documents, sync ChromaDB, and build the BM25 index (once)."""
    return load_documents(config.DATA_DIR)


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
def init_chat_state() -> None:
    """Ensure all session-state keys exist on first run."""
    defaults = {
        "messages": [],
        "last_matched_docs": None,
        "last_bm25_id": None,
        "last_emb_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_message_history() -> None:
    """Show all prior chat messages (with their retrieved context)."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("matched_docs"):
                render_context_expander("📄 Retrieved Context", message["matched_docs"], max_chars=400)


def render_context_expander(title: str, matched_docs: dict, max_chars: int) -> None:
    """Render a collapsible panel listing matched documents and their text."""
    with st.expander(title):
        for file_id, text in matched_docs.items():
            st.markdown(f"**`{file_id}`**")
            preview = text[:max_chars] + ("..." if len(text) > max_chars else "")
            st.code(preview)


def render_sample_queries() -> str | None:
    """Show clickable sample queries when the chat is empty."""
    if st.session_state.messages:
        return None

    st.markdown("### Try asking:")
    cols = st.columns(len(SAMPLE_QUERIES))
    for i, sample in enumerate(SAMPLE_QUERIES):
        if cols[i].button(sample, use_container_width=True):
            return sample
    return None


def render_assistant_response(prompt: str) -> None:
    """Retrieve documents and stream the LLM answer for the current prompt."""
    index = get_document_index()

    with st.spinner("🔍 Retrieving relevant documents..."):
        result = retrieve_hybrid(index, prompt)
        st.session_state.last_matched_docs = result.matched_contexts
        st.session_state.last_bm25_id = result.bm25_id
        st.session_state.last_emb_id = result.emb_id

    with st.chat_message("assistant"):
        if not result.matched_contexts:
            st.warning(NO_MATCH_RESPONSE)
            full_response = NO_MATCH_RESPONSE
        else:
            render_match_caption(result)
            full_response = stream_answer_with_cursor(prompt, result)
            render_context_expander(
                "📄 View Retrieved Context Sent to LLM",
                result.matched_contexts,
                max_chars=600,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "matched_docs": result.matched_contexts if result.matched_contexts else None,
            }
        )


def render_match_caption(result) -> None:
    """Show which channel matched which document (BM25 / Embedding)."""
    match_info = []
    if result.bm25_id:
        match_info.append(f"🔤 BM25: `{result.bm25_id}`")
    if result.emb_id:
        match_info.append(f"🧠 Embedding: `{result.emb_id}`")
    if match_info:
        st.caption(" | ".join(match_info))


def stream_answer_with_cursor(prompt: str, result) -> str:
    """Stream the answer token-by-token with a typing cursor effect."""
    placeholder = st.empty()
    full_response = ""

    # Exclude the just-added user message — the prompt already contains it.
    history_for_prompt = st.session_state.messages[:-1]

    for chunk in stream_answer(prompt, result.matched_contexts, history_for_prompt):
        full_response += chunk
        placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)
    return full_response


def render_sidebar(index) -> None:
    """Sidebar: document list, last matches, and action buttons."""
    with st.sidebar:
        st.header("📚 Document Collection")
        st.write(f"Indexed files: **{len(index.file_ids)}**")

        if st.session_state.last_matched_docs:
            st.markdown("---")
            st.subheader("🎯 Last Matched Documents")
            if st.session_state.last_bm25_id:
                st.info(f"🔤 BM25 Match: `{st.session_state.last_bm25_id}`")
            if st.session_state.last_emb_id:
                st.success(f"🧠 ChromaDB Match: `{st.session_state.last_emb_id}`")

        st.markdown("---")
        st.write("**All Indexed Files:**")
        for file_id in index.file_ids:
            st.text(f"• {file_id}")

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_matched_docs = None
            st.session_state.last_bm25_id = None
            st.session_state.last_emb_id = None
            st.rerun()

        if st.button("🔄 Re-index Files"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
init_chat_state()
index = get_document_index()

st.title("💬 Hybrid RAG Chat (BM25 + ChromaDB + Ollama)")
st.caption("Ask questions about the indexed documents. The system retrieves relevant context and streams answers.")

render_message_history()
sample_prompt = render_sample_queries()
prompt = st.chat_input("Ask a question about the documents...") or sample_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    render_assistant_response(prompt)

# Sidebar is rendered last so it always shows the latest matches.
render_sidebar(index)