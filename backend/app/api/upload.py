"""PDF upload endpoint."""

import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends

from app.api.dependencies import get_session_id
from app.api.services.parser import parse_pdf
from app.api.services.retrieval import get_base_ensemble_retriever
from app.api.services.vector_store import chunk_and_store

router = APIRouter(prefix="/api", tags=["upload"])


def _log(msg: str) -> None:
    print(f"[Saarthi] {msg}", flush=True)


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id)
):
    """
    Accept a PDF upload, parse it with LlamaParse,
    embed chunks with BGE-M3, and store in Qdrant.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    size_kb = (file.size or 0) / 1024
    _log(f"[SESSION] Uploading document for session={session_id}")
    _log(f"Upload started: {file.filename} ({size_kb:.1f} KB)")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        _log("Parsing PDF with LlamaParse...")
        documents = parse_pdf(tmp_path)
        
        # Replace the temporary file path in metadata with the actual uploaded filename
        for doc in documents:
            doc.metadata["source"] = file.filename
            
        _log(f"Parsed into {len(documents)} document(s)")

        _log("Chunking, embedding (BGE-M3), storing in Qdrant + BM25...")
        chunk_count = chunk_and_store(documents, session_id)
        _log(f"Stored {chunk_count} chunk(s) successfully")

        get_base_ensemble_retriever.cache_clear()
        _log("Retriever cache cleared")
    finally:
        os.unlink(tmp_path)

    return {
        "message": "PDF processed and stored successfully.",
        "filename": file.filename,
        "chunks_stored": chunk_count,
    }
