import pytest

from src.ingestion import load_pdf, split_documents
from src.retriever import HybridRetriever


@pytest.fixture
def loaded(store, embedder, sample_pdf):
    chunks = split_documents(load_pdf(sample_pdf))
    store.add_chunks(chunks, embedder.embed([c.page_content for c in chunks]), "technova_faq_sample.pdf")
    return store, len(chunks)


def test_adding_same_file_twice_does_not_duplicate(loaded, embedder, sample_pdf):
    store, n = loaded
    assert store.count() == n
    chunks = split_documents(load_pdf(sample_pdf))
    added, removed = store.add_chunks(
        chunks, embedder.embed([c.page_content for c in chunks]), "technova_faq_sample.pdf"
    )
    assert (added, removed) == (n, n)
    assert store.count() == n


def test_list_and_delete_source(loaded):
    store, n = loaded
    assert store.list_sources() == {"technova_faq_sample.pdf": n}
    assert store.delete_source("technova_faq_sample.pdf") == n
    assert store.count() == 0
    assert store.delete_source("technova_faq_sample.pdf") == 0


def test_collection_uses_cosine_distance(store):
    assert store.collection.metadata.get("hnsw:space") == "cosine"


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_retrieves_the_right_chunk_with_correct_citation(loaded, embedder, mode):
    store, _ = loaded
    retriever = HybridRetriever(embedder, store, mode=mode)
    out = retriever.retrieve("How many paid leaves does an employee get per year?", top_k=3)
    top = out["chunks"][0]
    assert "24 paid leaves" in top["text"]
    assert top["source"] == "technova_faq_sample.pdf"
    assert top["page"] == 1


def test_keyword_only_hit_keeps_real_metadata(loaded, embedder):
    store, _ = loaded
    retriever = HybridRetriever(embedder, store, mode="bm25")
    out = retriever.retrieve("AES-256", top_k=3)
    assert out["chunks"], "BM25 should find the exact keyword"
    assert all(c["source"] == "technova_faq_sample.pdf" for c in out["chunks"])
    assert all(isinstance(c["page"], int) for c in out["chunks"])


def test_bm25_ignores_chunks_with_no_matching_word(loaded, embedder):
    store, _ = loaded
    retriever = HybridRetriever(embedder, store, mode="bm25")
    assert retriever.retrieve("quantum entanglement spaceship", top_k=4)["chunks"] == []


def test_unrelated_question_has_low_similarity(loaded, embedder):
    store, _ = loaded
    retriever = HybridRetriever(embedder, store)
    related = retriever.retrieve("How many paid leaves per year?")["best_similarity"]
    unrelated = retriever.retrieve("quantum entanglement spaceship")["best_similarity"]
    assert unrelated < related
    assert unrelated < 0.1


def test_empty_store_and_blank_query(store, embedder):
    retriever = HybridRetriever(embedder, store)
    assert retriever.retrieve("anything") == {"chunks": [], "best_similarity": 0.0}


def test_invalid_mode(store, embedder):
    with pytest.raises(ValueError):
        HybridRetriever(embedder, store, mode="magic")
