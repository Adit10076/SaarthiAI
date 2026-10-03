"""RAG query endpoint."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_session_id
from app.api.services.rag import query as rag_query

router = APIRouter(prefix="/api", tags=["query"])


def _log(msg: str) -> None:
    print(f"[Saarthi] {msg}", flush=True)


class QueryRequest(BaseModel):
    """Incoming question from the frontend."""

    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    """RAG answer with source references."""

    answer: str
    sources: list[str]


@router.post("/query", response_model=QueryResponse)
async def ask_question(
    body: QueryRequest,
    session_id: str = Depends(get_session_id)
):
    """
    Retrieve relevant chunks from Qdrant and generate
    an answer using the Groq LLM.
    """
    _log(f"[SESSION] Retrieving documents for session={session_id}")
    _log(f"Question: {body.question}")
    try:
        result = rag_query(body.question, session_id)
    except Exception as exc:
        _log(f"Query FAILED: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _log(
        f"Answer ready | sources={len(result['sources'])} | "
        f"answer_length={len(result['answer'])}"
    )
    return QueryResponse(**result)
