"""Central settings, read from environment variables (and a local .env file)."""

import os

from dotenv import load_dotenv

load_dotenv()


def _clean_key(value):
    """Return the key, or "" if it is empty or still a placeholder like 'your_key_here'."""
    v = (value or "").strip().strip("\"'").strip()
    if not v or v.lower().startswith(("your", "paste")):
        return ""
    return v


def _get_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    def __init__(self):
        # --- LLM ---
        self.GROQ_API_KEY = _clean_key(os.getenv("GROQ_API_KEY"))
        self.OPENAI_API_KEY = _clean_key(os.getenv("OPENAI_API_KEY"))
        # auto = use Groq if its key is set, otherwise OpenAI
        self.LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
        self.GROQ_MODEL = (os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b").strip()
        self.OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        self.LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 1500)

        # --- Embeddings and storage ---
        self.EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2").strip()
        self.CHROMA_PERSIST_DIR = (os.getenv("CHROMA_PERSIST_DIR") or "data/vector_store").strip()
        self.UPLOAD_DIR = (os.getenv("UPLOAD_DIR") or "data/uploads").strip()

        # --- Chunking and retrieval ---
        self.CHUNK_SIZE = _get_int("CHUNK_SIZE", 500)
        self.CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 50)
        self.TOP_K = _get_int("TOP_K", 4)
        # If even the best chunk is less similar than this, we answer "not found"
        # without calling the LLM. Tune it for your documents.
        self.MIN_SIMILARITY = _get_float("MIN_SIMILARITY", 0.25)

        # --- Uploads ---
        self.MAX_UPLOAD_MB = _get_float("MAX_UPLOAD_MB", 10)


settings = Settings()
