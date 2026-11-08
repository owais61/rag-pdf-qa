import os

import pytest
from fastapi.testclient import TestClient

from app import app
from src.config import settings
from src.generator import LLMCallError, LLMUnavailableError
from src.service import get_service


@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def _upload(client, name, data, content_type="application/pdf"):
    return client.post("/upload", files={"file": (name, data, content_type)})


def _pdf_bytes(sample_pdf):
    with open(sample_pdf, "rb") as f:
        return f.read()


def test_home_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == 0


def test_upload_and_query_flow(client, sample_pdf):
    r = _upload(client, "technova_faq_sample.pdf", _pdf_bytes(sample_pdf))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success" and body["chunks_indexed"] > 0
    assert body["replaced_existing"] is False

    r = client.post("/query", json={"query": "How many paid leaves per year?", "top_k": 3})
    assert r.status_code == 200
    out = r.json()
    assert set(out) == {"answer", "answered", "sources", "usage"}
    assert out["sources"][0]["file"] == "technova_faq_sample.pdf"
    assert out["sources"][0]["page"] == 1
    assert "score" not in out["sources"][0]  # internal numbers are not exposed

    docs = client.get("/documents").json()["documents"]
    assert docs[0]["file"] == "technova_faq_sample.pdf"


def test_uploading_same_file_twice_reports_replacement(client, sample_pdf):
    data = _pdf_bytes(sample_pdf)
    first = _upload(client, "a.pdf", data).json()
    second = _upload(client, "a.pdf", data).json()
    assert second["replaced_existing"] is True
    assert second["chunks_indexed"] == first["chunks_indexed"]
    assert client.get("/health").json()["chunks"] == first["chunks_indexed"]


def test_path_traversal_filename_is_neutralised(client, sample_pdf):
    r = _upload(client, "../../evil.pdf", _pdf_bytes(sample_pdf))
    assert r.status_code == 200 and r.json()["file"] == "evil.pdf"
    assert os.path.exists(os.path.join(settings.UPLOAD_DIR, "evil.pdf"))
    assert not os.path.exists(os.path.join(settings.UPLOAD_DIR, "..", "..", "evil.pdf"))


def test_non_pdf_extension_rejected(client):
    assert _upload(client, "notes.txt", b"hello").status_code == 400


def test_fake_pdf_rejected_and_removed(client):
    r = _upload(client, "fake.pdf", b"not really a pdf")
    assert r.status_code == 400
    assert not os.path.exists(os.path.join(settings.UPLOAD_DIR, "fake.pdf"))


def test_oversized_upload_rejected(client, sample_pdf, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0.001)  # about 1 KB
    r = _upload(client, "big.pdf", _pdf_bytes(sample_pdf))
    assert r.status_code == 413
    assert not os.path.exists(os.path.join(settings.UPLOAD_DIR, "big.pdf"))


def test_pdf_without_text_returns_422(client, tmp_path):
    from pypdf import PdfWriter

    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with open(blank, "wb") as f:
        writer.write(f)
    r = _upload(client, "blank.pdf", blank.read_bytes())
    assert r.status_code == 422
    assert "No text" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "   "}, {"query": "x" * 1001}, {"query": "ok", "top_k": 99},
     {"query": "ok", "top_k": 0}, {}],
)
def test_query_validation(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_query_without_api_key_gives_503(client, service, monkeypatch):
    class NoKey:
        def ask(self, *a, **k):
            raise LLMUnavailableError("No LLM API key found.")

    app.dependency_overrides[get_service] = lambda: NoKey()
    r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 503 and "API key" in r.json()["detail"]


def test_query_llm_failure_gives_502(client):
    class Broken:
        def ask(self, *a, **k):
            raise LLMCallError("NotFoundError: model_not_found")

    app.dependency_overrides[get_service] = lambda: Broken()
    r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 502 and "model_not_found" in r.json()["detail"]


def test_delete_document(client, sample_pdf):
    _upload(client, "a.pdf", _pdf_bytes(sample_pdf))
    assert os.path.exists(os.path.join(settings.UPLOAD_DIR, "a.pdf"))

    r = client.delete("/documents/a.pdf")
    assert r.status_code == 200 and r.json()["chunks_removed"] > 0
    assert not os.path.exists(os.path.join(settings.UPLOAD_DIR, "a.pdf"))
    assert client.delete("/documents/a.pdf").status_code == 404
    assert client.delete("/documents/notes.txt").status_code == 400
