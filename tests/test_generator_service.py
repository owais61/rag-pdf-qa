import pytest

from src import generator
from src.config import settings
from src.generator import (
    NO_ANSWER,
    LLMCallError,
    LLMUnavailableError,
    generate_answer,
    get_llm,
    resolve_provider,
)
from tests.conftest import FakeLLM

CHUNKS = [
    {"text": "Every employee gets 24 paid leaves per year.", "source": "a.pdf", "page": 1},
    {"text": "Sick leave is 12 days.", "source": "a.pdf", "page": 1},
]


# ---------------- generator ----------------
def test_generate_answer_builds_numbered_context_and_sources():
    llm = FakeLLM()
    out = generate_answer("How many leaves?", CHUNKS, llm)
    system, human = llm.calls[0]
    assert "ONLY" in system[1]
    assert "[1] (file: a.pdf, page: 1)" in human[1] and "[2]" in human[1]
    assert "How many leaves?" in human[1]
    assert out["answered"] is True
    assert [s["ref"] for s in out["sources"]] == [1, 2]  # citation [2] -> sources[1]
    assert out["usage"] == {"input_tokens": 100, "output_tokens": 20}


def test_think_tags_are_removed():
    out = generate_answer("q", CHUNKS, FakeLLM("<think>secret reasoning</think>Final answer [1]."))
    assert out["answer"] == "Final answer [1]."


def test_no_chunks_means_no_llm_call():
    llm = FakeLLM()
    out = generate_answer("q", [], llm)
    assert out["answer"] == NO_ANSWER and out["answered"] is False and not llm.calls


def test_model_saying_not_found_is_marked_unanswered():
    out = generate_answer("q", CHUNKS, FakeLLM(NO_ANSWER))
    assert out["answered"] is False


def test_empty_llm_answer_raises():
    with pytest.raises(LLMCallError):
        generate_answer("q", CHUNKS, FakeLLM("   "))


def test_llm_exception_is_wrapped():
    class Boom:
        def invoke(self, _):
            raise RuntimeError("model_not_found")

    with pytest.raises(LLMCallError, match="model_not_found"):
        generate_answer("q", CHUNKS, Boom())


def test_provider_selection(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    assert resolve_provider() is None
    with pytest.raises(LLMUnavailableError):
        get_llm()

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    assert resolve_provider() == "openai"  # only OpenAI key set -> OpenAI is used

    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test")
    assert resolve_provider() == "groq"  # both set -> Groq first

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    assert resolve_provider() == "openai"

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    assert resolve_provider() is None  # asked for OpenAI but no key


def test_get_llm_uses_model_name_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-20b")
    llm = get_llm()
    assert llm.model_name == "openai/gpt-oss-20b"


def test_placeholder_keys_are_treated_as_missing():
    from src.config import _clean_key

    assert _clean_key("your_groq_api_key_here") == ""
    assert _clean_key("paste-your-key") == ""
    assert _clean_key('  "gsk_real"  ') == "gsk_real"
    assert _clean_key(None) == ""


# ---------------- service ----------------
def test_ingest_twice_replaces_instead_of_duplicating(service, sample_pdf):
    first = service.ingest_pdf(sample_pdf)
    second = service.ingest_pdf(sample_pdf)
    assert first["replaced_existing"] is False
    assert second["replaced_existing"] is True
    assert service.store.count() == first["chunks_indexed"] == second["chunks_indexed"]
    assert service.list_documents() == [
        {"file": "technova_faq_sample.pdf", "chunks": first["chunks_indexed"]}
    ]


def test_ask_returns_answer_with_sources(service, llm, sample_pdf):
    service.ingest_pdf(sample_pdf)
    out = service.ask("How many paid leaves does an employee get per year?")
    assert out["answered"] is True
    assert out["sources"] and out["sources"][0]["file"] == "technova_faq_sample.pdf"
    assert len(llm.calls) == 1
    assert "24 paid leaves" in llm.calls[0][1][1]  # the right chunk was sent to the LLM


def test_unrelated_question_skips_the_llm(service, llm, sample_pdf):
    service.ingest_pdf(sample_pdf)
    out = service.ask("quantum entanglement spaceship")
    assert out["answer"] == NO_ANSWER and out["answered"] is False and out["sources"] == []
    assert llm.calls == []


def test_ask_with_no_documents(service, llm):
    out = service.ask("anything at all")
    assert out["answered"] is False and llm.calls == []


def test_ask_without_api_key_raises_clear_error(embedder, store, monkeypatch):
    from src.service import RAGService

    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    service = RAGService(embedder=embedder, store=store)  # no llm injected
    with pytest.raises(LLMUnavailableError):
        service.ask("hello")


def test_delete_document(service, sample_pdf):
    service.ingest_pdf(sample_pdf)
    assert service.delete_document("technova_faq_sample.pdf") > 0
    assert service.list_documents() == []
    assert service.delete_document("technova_faq_sample.pdf") == 0


def test_status(service, sample_pdf):
    service.ingest_pdf(sample_pdf)
    status = service.status()
    assert status["documents"] == 1 and status["chunks"] > 0


def test_module_exports():
    assert generator.NO_ANSWER
