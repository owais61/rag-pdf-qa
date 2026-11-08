"""Shared test helpers.

The tests never download a model or call an LLM API. HashingEmbedder is a tiny stand-in
for the real embedding model (similar words -> similar vectors), and FakeLLM records
what it was asked.
"""

import hashlib
import math
import os

import pytest

from src.config import settings
from src.retriever import tokenize
from src.service import RAGService
from src.vectorstore import VectorStoreManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PDF = os.path.join(ROOT, "docs", "sample", "technova_faq_sample.pdf")


class HashingEmbedder:
    dim = 512

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):  # ignores filler words like "the", "does"
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class FakeResponse:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage


class FakeLLM:
    def __init__(self, content="The answer is 24 paid leaves per year [1].", usage=None):
        self.content = content
        self.usage = usage or {"input_tokens": 100, "output_tokens": 20}
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeResponse(self.content, self.usage)


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path):
    return VectorStoreManager(persist_dir=str(tmp_path / "chroma"), collection_name="test")


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def service(embedder, store, llm, monkeypatch, tmp_path):
    # The hashing embedder gives lower similarity numbers than a real model.
    monkeypatch.setattr(settings, "MIN_SIMILARITY", 0.1)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    return RAGService(embedder=embedder, store=store, llm=llm)


@pytest.fixture
def sample_pdf():
    return SAMPLE_PDF
