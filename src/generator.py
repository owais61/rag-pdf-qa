"""Step 4 of RAG: give the retrieved chunks to an LLM and let it write the answer."""

import re
from typing import Any, Dict, List

from src.config import settings
from src.utils import collapse_whitespace

NO_ANSWER = "I could not find this in the uploaded documents."

SYSTEM_PROMPT = (
    "You answer questions about the user's documents.\n"
    "Rules:\n"
    "- Use ONLY the numbered context blocks provided. Do not use outside knowledge.\n"
    f'- If the context does not contain the answer, reply exactly: "{NO_ANSWER}"\n'
    "- Be concise and accurate.\n"
    "- After each fact, cite the block it came from, like [1] or [2]."
)


class LLMUnavailableError(RuntimeError):
    """No API key is configured (or the chosen provider has no key)."""


class LLMCallError(RuntimeError):
    """The LLM provider rejected or failed the request."""


def resolve_provider():
    """Return 'groq', 'openai' or None, based on LLM_PROVIDER and the available keys."""
    choice = settings.LLM_PROVIDER
    if choice == "groq":
        return "groq" if settings.GROQ_API_KEY else None
    if choice == "openai":
        return "openai" if settings.OPENAI_API_KEY else None
    if settings.GROQ_API_KEY:
        return "groq"
    if settings.OPENAI_API_KEY:
        return "openai"
    return None


def get_llm():
    provider = resolve_provider()
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    raise LLMUnavailableError(
        "No LLM API key found. Put GROQ_API_KEY or OPENAI_API_KEY in your .env file "
        "and restart the app."
    )


def build_context(chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (file: {c['source']}, page: {c['page']})\n{c['text']}")
    return "\n\n".join(blocks)


def _text_of(content) -> str:
    if isinstance(content, list):  # some providers return a list of parts
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    text = str(content or "")
    # remove <think>...</think> blocks that some reasoning models include
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_answer(query: str, chunks: List[Dict[str, Any]], llm) -> Dict[str, Any]:
    """Ask the LLM to answer using only these chunks.

    Returns {"answer", "sources", "usage", "answered"}. `sources` has one entry per
    context block, so a citation like [2] in the answer maps to sources[1].
    """
    if not chunks:
        return {"answer": NO_ANSWER, "sources": [], "usage": None, "answered": False}

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{build_context(chunks)}\n\nQuestion: {query}"),
    ]
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        raise LLMCallError(f"{type(exc).__name__}: {str(exc)[:300]}") from exc

    answer = _text_of(getattr(response, "content", ""))
    if not answer:
        raise LLMCallError(
            "The model returned an empty answer. Try again, raise LLM_MAX_TOKENS, "
            "or use a different model."
        )

    sources = [
        {
            "ref": i,
            "file": c["source"],
            "page": c["page"],
            "snippet": collapse_whitespace(c["text"])[:200],
        }
        for i, c in enumerate(chunks, start=1)
    ]

    usage = None
    meta = getattr(response, "usage_metadata", None)
    if meta:
        usage = {
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
        }

    return {
        "answer": answer,
        "sources": sources,
        "usage": usage,
        "answered": answer.strip() != NO_ANSWER,
    }
