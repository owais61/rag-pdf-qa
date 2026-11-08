"""Smoke-test the Streamlit UI with Streamlit's built-in AppTest (no browser needed)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT


@pytest.fixture
def app(service, monkeypatch):
    monkeypatch.chdir(ROOT)  # ui.py looks for docs/sample/... relative to the project root
    monkeypatch.setattr("src.service.get_service", lambda: service)
    return AppTest.from_file(os.path.join(ROOT, "ui.py"), default_timeout=30)


def _button(app, label):
    return next(b for b in app.button if b.label == label)


def test_ui_loads_without_errors(app):
    app.run()
    assert not app.exception
    assert app.title[0].value.endswith("Ask your PDFs")


def test_asking_before_uploading_shows_hint(app):
    app.run()
    app.chat_input[0].set_value("hello?").run()
    assert not app.exception
    assert any("upload a PDF first" in m.value for m in app.markdown)


def test_sample_pdf_then_question_shows_answer(app, llm):
    app.run()
    _button(app, "Load sample PDF").click().run()
    assert not app.exception
    assert any("Indexed technova_faq_sample.pdf" in s.value for s in app.sidebar.success)

    app.chat_input[0].set_value("How many paid leaves does an employee get per year?").run()
    assert not app.exception
    assert any("24 paid leaves" in m.value for m in app.markdown)
    assert len(llm.calls) == 1
