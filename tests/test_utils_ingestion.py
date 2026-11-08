import pytest
from pypdf import PdfWriter

from src.ingestion import load_pdf, split_documents
from src.utils import safe_filename


def test_safe_filename_blocks_path_traversal():
    assert safe_filename("../../evil.pdf") == "evil.pdf"
    assert safe_filename("C:\\Users\\me\\notes.pdf") == "notes.pdf"
    assert safe_filename("/etc/passwd.pdf") == "passwd.pdf"


def test_safe_filename_accepts_uppercase_extension_and_cleans_chars():
    assert safe_filename("My Notes (1).PDF") == "My Notes _1_.PDF"


@pytest.mark.parametrize("bad", ["notes.txt", "", None, ".pdf", "image.png"])
def test_safe_filename_rejects_non_pdf(bad):
    with pytest.raises(ValueError):
        safe_filename(bad)


def test_load_pdf_returns_pages_with_metadata(sample_pdf):
    docs = load_pdf(sample_pdf)
    assert [d.metadata["page"] for d in docs] == [1, 2]  # pages start at 1
    assert all(d.metadata["source"] == "technova_faq_sample.pdf" for d in docs)
    assert "24 paid leaves" in docs[0].page_content


def test_split_documents_respects_chunk_size_and_keeps_metadata(sample_pdf):
    chunks = split_documents(load_pdf(sample_pdf), chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 2
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all("source" in c.metadata and "page" in c.metadata for c in chunks)


def test_load_pdf_rejects_pdf_without_text(tmp_path):
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(blank, "wb") as f:
        writer.write(f)
    with pytest.raises(ValueError, match="No text"):
        load_pdf(str(blank))


def test_load_pdf_rejects_fake_pdf(tmp_path):
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"this is not a pdf")
    with pytest.raises(ValueError):
        load_pdf(str(fake))


def test_load_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("does_not_exist.pdf")
