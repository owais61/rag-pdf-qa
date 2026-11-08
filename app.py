"""FastAPI backend.  Run:  uvicorn app:app --reload --port 8000"""

import os
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.generator import LLMCallError, LLMUnavailableError
from src.service import RAGService, get_service
from src.utils import safe_filename

app = FastAPI(
    title="RAG PDF Q&A API",
    version="1.0.0",
    description="Upload PDFs, then ask questions. Answers come with source citations.",
)


# ---------- request / response models ----------
class QueryRequest(BaseModel):
    query: str = Field(..., max_length=1000, examples=["How many paid leaves per year?"])
    top_k: Optional[int] = Field(None, ge=1, le=10, description="How many chunks to retrieve")

    @field_validator("query")
    @classmethod
    def not_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class Source(BaseModel):
    ref: int
    file: str
    page: Optional[int] = None
    snippet: str


class Usage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    answered: bool
    sources: List[Source]
    usage: Optional[Usage] = None


# ---------- endpoints ----------
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(svc: RAGService = Depends(get_service)):
    return svc.status()


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...), svc: RAGService = Depends(get_service)):
    try:
        filename = safe_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    path = os.path.join(settings.UPLOAD_DIR, filename)

    limit = int(settings.MAX_UPLOAD_MB * 1024 * 1024)
    size = 0
    with open(path, "wb") as out:
        while True:
            block = file.file.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > limit:
                out.close()
                os.remove(path)
                raise HTTPException(
                    status_code=413, detail=f"File is larger than {settings.MAX_UPLOAD_MB:g} MB."
                )
            out.write(block)

    with open(path, "rb") as check:
        if check.read(5) != b"%PDF-":
            os.remove(path)
            raise HTTPException(status_code=400, detail="This file is not a valid PDF.")

    try:
        result = svc.ingest_pdf(path)
    except ValueError as exc:
        os.remove(path)
        raise HTTPException(status_code=422, detail=str(exc))

    return {"status": "success", **result}


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest, svc: RAGService = Depends(get_service)):
    try:
        return svc.ask(req.query, top_k=req.top_k)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except LLMCallError as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}")


@app.get("/documents")
def list_documents(svc: RAGService = Depends(get_service)):
    return {"documents": svc.list_documents()}


@app.delete("/documents/{filename}")
def delete_document(filename: str, svc: RAGService = Depends(get_service)):
    try:
        filename = safe_filename(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    removed = svc.delete_document(filename)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    path = os.path.join(settings.UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    return {"status": "deleted", "file": filename, "chunks_removed": removed}
