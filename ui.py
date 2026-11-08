"""Streamlit chat UI.  Run:  streamlit run ui.py"""

import os

import streamlit as st

from src.config import settings
from src.generator import LLMCallError, LLMUnavailableError
from src.service import get_service
from src.utils import safe_filename

SAMPLE_PDF = os.path.join("docs", "sample", "technova_faq_sample.pdf")

st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄")
svc = get_service()


def index_pdf(filename: str, data: bytes):
    """Validate, save and index an uploaded PDF. Shows the result in the sidebar."""
    try:
        name = safe_filename(filename)
    except ValueError as exc:
        st.error(str(exc))
        return
    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is larger than {settings.MAX_UPLOAD_MB:g} MB.")
        return
    if not data.startswith(b"%PDF-"):
        st.error("This file is not a valid PDF.")
        return

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    path = os.path.join(settings.UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    try:
        with st.spinner("Reading and indexing the PDF..."):
            result = svc.ingest_pdf(path)
        st.success(f"Indexed {result['file']} ({result['chunks_indexed']} chunks)")
    except ValueError as exc:
        os.remove(path)
        st.error(str(exc))


def show_sources(sources, usage):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"**[{s['ref']}] {s['file']}, page {s['page']}**")
                st.caption(s["snippet"] + "...")
    if usage and usage.get("input_tokens") is not None:
        st.caption(f"Tokens used: {usage['input_tokens']} in / {usage['output_tokens']} out")


# ---------------- sidebar: documents ----------------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.button("Index PDF", type="primary"):
        index_pdf(uploaded.name, uploaded.getvalue())

    if os.path.exists(SAMPLE_PDF) and st.button("Load sample PDF"):
        with open(SAMPLE_PDF, "rb") as f:
            index_pdf(os.path.basename(SAMPLE_PDF), f.read())

    st.divider()
    documents = svc.list_documents()
    if not documents:
        st.caption("No documents yet.")
    for doc in documents:
        left, right = st.columns([4, 1])
        left.write(f"{doc['file']}  \n{doc['chunks']} chunks")
        if right.button("✕", key=f"delete_{doc['file']}", help="Remove this document"):
            svc.delete_document(doc["file"])
            path = os.path.join(settings.UPLOAD_DIR, doc["file"])
            if os.path.exists(path):
                os.remove(path)
            st.rerun()

    st.divider()
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, settings.TOP_K)
    status = svc.status()
    if status["llm_configured"]:
        st.caption(f"LLM: {status['llm_provider']} / {status['llm_model']}")
    else:
        st.warning("No API key found. Add GROQ_API_KEY or OPENAI_API_KEY to .env and restart.")

# ---------------- main: chat ----------------
st.title("📄 Ask your PDFs")
st.caption("Answers come only from your uploaded documents, with sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources"), message.get("usage"))

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not documents:
            reply = {"answer": "Please upload a PDF first (sidebar).", "sources": [], "usage": None}
        else:
            try:
                with st.spinner("Thinking..."):
                    reply = svc.ask(question, top_k=top_k)
            except LLMUnavailableError as exc:
                reply = {"answer": f"⚠️ {exc}", "sources": [], "usage": None}
            except LLMCallError as exc:
                reply = {"answer": f"⚠️ The LLM request failed: {exc}", "sources": [], "usage": None}
        st.markdown(reply["answer"])
        show_sources(reply["sources"], reply["usage"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply["answer"],
            "sources": reply["sources"],
            "usage": reply["usage"],
        }
    )
