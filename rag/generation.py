"""Answer generation: prompt construction and streaming from Ollama."""
import json
from typing import Dict, Iterator, List, Optional

import requests

from . import config

# How many prior messages (last N) are included for conversational context.
SYSTEM_INSTRUCTION = """You are a helpful assistant. Use the provided contexts from retrieved documents to answer the question.
If information spans across multiple documents, combine and summarize it effectively.
CRITICAL: Do not hallucinate. Only use the information provided in the contexts. Make sure
there are at least 3 sentences from the contexts to answer the question."""


def _format_contexts(context_dict: Dict[str, str]) -> str:
    """Render the retrieved documents as a single labelled block of text."""
    formatted = [
        f"--- DOCUMENT: {file_id} ---\n{text}"
        for file_id, text in context_dict.items()
    ]
    return "\n\n".join(formatted)


def _format_history(chat_history: Optional[List[dict]]) -> str:
    """Render recent conversation history as a labelled block of text."""
    if not chat_history:
        return "(No prior conversation)"

    lines = []
    for message in chat_history[-config.HISTORY_TURNS:]:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")

    return "\n".join(lines) if lines else "(No prior conversation)"


def build_prompt(query: str, context_dict: Dict[str, str], chat_history: Optional[List[dict]] = None) -> str:
    """Assemble the full prompt sent to the LLM."""
    return f"""{SYSTEM_INSTRUCTION}

[CONVERSATION HISTORY]
{_format_history(chat_history)}

[CONTEXTS]
{_format_contexts(context_dict)}

[USER QUESTION]
{query}
"""


def stream_answer(query: str, context_dict: Dict[str, str], chat_history: Optional[List[dict]] = None) -> Iterator[str]:
    """Stream the model's response from Ollama, one chunk at a time.

    Yields plain text chunks as they arrive. If Ollama is unreachable,
    yields a single error message instead of raising.
    """
    prompt = build_prompt(query, context_dict, chat_history)

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
        },
    }

    try:
        with requests.post(
            config.OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=config.LLM_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    yield chunk.get("response", "")
    except Exception as exc:
        yield f"Error connecting to Ollama at {config.OLLAMA_URL}: {exc}"