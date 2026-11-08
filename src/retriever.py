"""Step 3 of RAG: find the chunks that are most likely to contain the answer.

Two searches are combined (hybrid search):
  * dense  - compares the meaning of the question with every chunk (embeddings)
  * bm25   - compares the actual words (keywords)
Their rankings are merged with Reciprocal Rank Fusion (RRF), which needs no
hand-tuned score weights.
"""

import re
from collections import defaultdict
from typing import Any, Dict

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on", "for",
    "and", "or", "by", "at", "as", "it", "this", "that", "with", "from", "what", "who",
    "how", "does", "do", "can", "i", "my", "me",
}

MODES = ("hybrid", "dense", "bm25")


def tokenize(text: str):
    return [t for t in re.findall(r"\w+", (text or "").lower()) if t not in STOPWORDS]


class HybridRetriever:
    RRF_K = 60  # standard constant from the RRF paper

    def __init__(self, embedder, store, mode: str = "hybrid"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.embedder = embedder
        self.store = store
        self.mode = mode
        self._ids, self._docs, self._metas = [], [], []
        self._bm25 = None
        self.refresh()

    def refresh(self):
        """Rebuild the in-memory keyword index. Call after documents are added/removed."""
        data = self.store.get_all()
        self._ids, self._docs, self._metas = data["ids"], data["documents"], data["metadatas"]
        tokenized = [tokenize(d) for d in self._docs]
        if any(tokenized):
            # BM25 cannot handle empty documents, so give them a dummy token.
            self._bm25 = BM25Okapi([t or ["__empty__"] for t in tokenized])
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Return {"chunks": [...], "best_similarity": float}.

        best_similarity is the cosine similarity of the closest chunk (0 to 1). The caller
        uses it to decide that the documents do not contain an answer at all.
        """
        total = self.store.count()
        if total == 0 or not (query or "").strip():
            return {"chunks": [], "best_similarity": 0.0}

        pool = min(total, max(top_k * 4, 20))

        # ---- dense (meaning) search ----
        dense_hits = []  # (id, text, meta, similarity)
        if self.mode in ("dense", "hybrid"):
            q_vec = self.embedder.embed([query])[0]
            res = self.store.collection.query(
                query_embeddings=[q_vec],
                n_results=pool,
                include=["documents", "metadatas", "distances"],
            )
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                dense_hits.append((cid, doc, meta, 1.0 - float(dist)))

        # ---- keyword (BM25) search ----
        bm25_hits = []  # positions in self._ids
        if self.mode in ("bm25", "hybrid") and self._bm25 is not None:
            tokens = tokenize(query)
            if tokens:
                scores = self._bm25.get_scores(tokens)
                for idx in np.argsort(scores)[::-1][:pool]:
                    if scores[idx] > 0:  # skip chunks that share no word with the query
                        bm25_hits.append(int(idx))

        # ---- Reciprocal Rank Fusion ----
        fused = defaultdict(float)
        info = {}
        for rank, (cid, doc, meta, sim) in enumerate(dense_hits):
            fused[cid] += 1.0 / (self.RRF_K + rank + 1)
            info[cid] = {"text": doc, "meta": meta or {}, "sim": sim}
        for rank, idx in enumerate(bm25_hits):
            cid = self._ids[idx]
            fused[cid] += 1.0 / (self.RRF_K + rank + 1)
            # keyword-only hit: keep the REAL metadata so citations stay correct
            info.setdefault(
                cid, {"text": self._docs[idx], "meta": self._metas[idx] or {}, "sim": None}
            )

        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        chunks = []
        for cid in ranked:
            item = info[cid]
            chunks.append(
                {
                    "id": cid,
                    "text": item["text"],
                    "source": item["meta"].get("source", "unknown"),
                    "page": item["meta"].get("page"),
                    "similarity": None if item["sim"] is None else round(item["sim"], 4),
                    "score": round(fused[cid], 5),
                }
            )

        best = max((h[3] for h in dense_hits), default=0.0)
        return {"chunks": chunks, "best_similarity": round(best, 4)}
