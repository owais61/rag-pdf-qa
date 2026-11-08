"""The RAG pipeline in one place. Both the FastAPI app and the Streamlit UI use this."""

import os
from functools import lru_cache
from typing import Any, Dict, List

from src.config import settings
from src.generator import NO_ANSWER, generate_answer, get_llm, resolve_provider
from src.ingestion import load_pdf, split_documents
from src.retriever import HybridRetriever
from src.vectorstore import EmbeddingManager, VectorStoreManager


class RAGService:
    """Heavy parts (embedding model, database, LLM client) are created on first use.

    You can pass your own embedder/store/llm, which is how the tests run without
    downloading a model or calling an API.
    """

    def __init__(self, embedder=None, store=None, llm=None):
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._retriever = None

    # ---- lazily created parts ----
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = EmbeddingManager()
        return self._embedder

    @property
    def store(self):
        if self._store is None:
            self._store = VectorStoreManager()
        return self._store

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = HybridRetriever(self.embedder, self.store)
        return self._retriever

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()  # raises LLMUnavailableError if no key
        return self._llm

    # ---- documents ----
    def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Read, chunk, embed and store one PDF. Re-ingesting a file replaces it."""
        source = os.path.basename(pdf_path)
        chunks = split_documents(load_pdf(pdf_path))
        embeddings = self.embedder.embed([c.page_content for c in chunks])
        added, removed = self.store.add_chunks(chunks, embeddings, source)
        self.retriever.refresh()
        return {"file": source, "chunks_indexed": added, "replaced_existing": removed > 0}

    def list_documents(self) -> List[Dict[str, Any]]:
        return [
            {"file": name, "chunks": n}
            for name, n in sorted(self.store.list_sources().items())
        ]

    def delete_document(self, filename: str) -> int:
        removed = self.store.delete_source(filename)
        self.retriever.refresh()
        return removed

    # ---- question answering ----
    def ask(self, query: str, top_k: int = None) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise ValueError("Question is empty.")
        top_k = top_k or settings.TOP_K

        llm = self.llm  # fail early (with a clear message) if no API key is set
        result = self.retriever.retrieve(query, top_k=top_k)

        if not result["chunks"] or result["best_similarity"] < settings.MIN_SIMILARITY:
            return {"answer": NO_ANSWER, "sources": [], "usage": None, "answered": False}
        return generate_answer(query, result["chunks"], llm)

    def status(self) -> Dict[str, Any]:
        provider = resolve_provider()
        model = {"groq": settings.GROQ_MODEL, "openai": settings.OPENAI_MODEL}.get(provider)
        return {
            "status": "ok",
            "documents": len(self.store.list_sources()),
            "chunks": self.store.count(),
            "llm_configured": provider is not None,
            "llm_provider": provider,
            "llm_model": model,
        }


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return RAGService()
