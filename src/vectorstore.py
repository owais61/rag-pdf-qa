"""Step 2 of RAG: turn chunks into embeddings and store them in ChromaDB."""

import hashlib
import os
from typing import Dict, List

import chromadb
from langchain_core.documents import Document

from src.config import settings

_BATCH = 500


class EmbeddingManager:
    """Wraps a sentence-transformers model. Vectors are normalized (length 1)."""

    def __init__(self, model_name: str = None):
        # Imported here so the app and tests start fast and do not need torch
        # until a real model is actually used.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vectors.tolist()


def _chunk_id(source: str, index: int) -> str:
    return hashlib.sha1(f"{source}:{index}".encode("utf-8")).hexdigest()


class VectorStoreManager:
    """ChromaDB collection using cosine distance (so similarity = 1 - distance)."""

    def __init__(self, persist_dir: str = None, collection_name: str = "rag_documents"):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def delete_source(self, source: str) -> int:
        """Remove every chunk that came from this file. Returns how many were removed."""
        found = self.collection.get(where={"source": source}, include=[])
        ids = found["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def add_chunks(self, chunks: List[Document], embeddings, source: str):
        """Store the chunks of one file. Re-adding the same file replaces the old chunks,
        so uploading it twice never creates duplicates.

        Returns (chunks_added, old_chunks_removed).
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        removed = self.delete_source(source)

        ids, docs, metas, vectors = [], [], [], []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            ids.append(_chunk_id(source, i))
            docs.append(chunk.page_content)
            metas.append(
                {"source": source, "page": int(chunk.metadata.get("page", 0)), "chunk": i}
            )
            vectors.append([float(x) for x in vec])

        for start in range(0, len(ids), _BATCH):
            end = start + _BATCH
            self.collection.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
                embeddings=vectors[start:end],
            )
        return len(ids), removed

    def list_sources(self) -> Dict[str, int]:
        """{filename: number_of_chunks}"""
        data = self.collection.get(include=["metadatas"])
        counts: Dict[str, int] = {}
        for meta in data["metadatas"] or []:
            name = (meta or {}).get("source", "unknown")
            counts[name] = counts.get(name, 0) + 1
        return counts

    def get_all(self):
        """All stored chunks (used to build the keyword/BM25 index)."""
        data = self.collection.get(include=["documents", "metadatas"])
        return {
            "ids": data["ids"] or [],
            "documents": data["documents"] or [],
            "metadatas": data["metadatas"] or [],
        }
