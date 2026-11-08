"""Step 1 of RAG: read a PDF and cut it into small chunks."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import settings


def load_pdf(pdf_path: str) -> List[Document]:
    """Return one Document per page that has text. Page numbers start at 1."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    name = os.path.basename(pdf_path)
    try:
        reader = PdfReader(pdf_path)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # corrupt, encrypted, not really a PDF, ...
        raise ValueError(f"Could not read this PDF ({type(exc).__name__}).") from exc

    docs = [
        Document(page_content=text, metadata={"source": name, "page": number})
        for number, text in enumerate(pages, start=1)
        if text
    ]
    if not docs:
        raise ValueError(
            "No text could be extracted. This looks like a scanned PDF (images only); "
            "OCR is not supported."
        )
    return docs


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """Split pages into overlapping chunks. Each chunk keeps its source and page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)
