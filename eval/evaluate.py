"""Measure how good the RETRIEVAL step is (no LLM, no API key, no cost).

For each question we know a phrase that the correct chunk must contain. A question is a
"hit@k" if one of the top-k retrieved chunks contains that phrase. We also report MRR
(mean reciprocal rank: 1.0 if the right chunk is first, 0.5 if second, and so on).

Run (uses the real embedding model, downloaded on first run):

    python -m eval.evaluate
    python -m eval.evaluate --pdf my.pdf --questions my_questions.json --ks 1 3 5

questions.json is a list of {"question": ..., "answer_contains": ...}.
"""

import argparse
import json
import os
import shutil
import tempfile

from src.ingestion import load_pdf, split_documents
from src.retriever import HybridRetriever
from src.utils import collapse_whitespace
from src.vectorstore import VectorStoreManager


def _norm(text):
    return collapse_whitespace(text).lower()


def run_eval(embedder, pdf_path, questions, ks=(1, 3, 5), modes=("dense", "bm25", "hybrid")):
    """Return {mode: {"hit@1": 0.8, ..., "mrr": 0.7, "n": 14}}."""
    tmp = tempfile.mkdtemp(prefix="rag_eval_")
    try:
        store = VectorStoreManager(persist_dir=tmp, collection_name="eval")
        chunks = split_documents(load_pdf(pdf_path))
        embeddings = embedder.embed([c.page_content for c in chunks])
        store.add_chunks(chunks, embeddings, os.path.basename(pdf_path))

        results = {}
        for mode in modes:
            retriever = HybridRetriever(embedder, store, mode=mode)
            hits = {k: 0 for k in ks}
            reciprocal_ranks = []
            for item in questions:
                needle = _norm(item["answer_contains"])
                got = retriever.retrieve(item["question"], top_k=max(ks))["chunks"]
                rank = next(
                    (i for i, c in enumerate(got, start=1) if needle in _norm(c["text"])), None
                )
                for k in ks:
                    if rank is not None and rank <= k:
                        hits[k] += 1
                reciprocal_ranks.append(1.0 / rank if rank else 0.0)

            n = len(questions)
            row = {f"hit@{k}": hits[k] / n for k in ks}
            row["mrr"] = sum(reciprocal_ranks) / n
            row["n"] = n
            results[mode] = row
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # ignore_errors: Windows may keep files locked


def print_table(results, ks):
    header = ["mode"] + [f"hit@{k}" for k in ks] + ["MRR"]
    print(" | ".join(f"{h:>8}" for h in header))
    print("-" * (11 * len(header)))
    for mode, row in results.items():
        cells = [mode] + [f"{row[f'hit@{k}']:.2f}" for k in ks] + [f"{row['mrr']:.2f}"]
        print(" | ".join(f"{c:>8}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--pdf", default="docs/sample/technova_faq_sample.pdf")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--json", help="optional path to save the results as JSON")
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    from src.vectorstore import EmbeddingManager  # loads the real model

    print(f"Evaluating {len(questions)} questions on {args.pdf} ...\n")
    results = run_eval(EmbeddingManager(), args.pdf, questions, ks=tuple(args.ks))
    print_table(results, args.ks)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
