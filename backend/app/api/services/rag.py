"""RAG retrieval chain: hybrid search → rerank → Groq response."""

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import settings
from app.api.services.retrieval import get_base_ensemble_retriever
from app.api.services.reranker import get_reranker_model
from app.api.services.advanced_retrieval import transform_query


def _log(msg: str) -> None:
    print(f"[Saarthi] {msg}", flush=True)

# System prompt for the RAG chain
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the provided context. "
            "If the context does not contain enough information, say so.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ]
)


def _format_docs(docs) -> str:
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


from app.api.services.llm import get_llm


def query(question: str, session_id: str) -> dict:
    """
    Run a RAG query using advanced retrieval techniques and return the answer.

    Returns:
        {"answer": str, "sources": list[str]}
    """
    _log(f"Starting advanced RAG query pipeline for: '{question}'")
    
    # 1. Apply Query Transformations (Rewrite, Multi-Query, HyDE)
    transformed = transform_query(question)
    rewritten_query = transformed["rewritten"]
    queries_to_run = [rewritten_query] + transformed["variations"] + [transformed["hyde_doc"]]
    
    # 2. Retrieve for all variations using the base ensemble retriever
    _log(f"Running retrieval for {len(queries_to_run)} queries (Original + Variations + HyDE)...")
    base_retriever = get_base_ensemble_retriever(session_id)
    
    all_docs = []
    for q in queries_to_run:
        # We use invoke to get the documents for each query
        docs = base_retriever.invoke(q)
        all_docs.extend(docs)
        
    # 3. Merge and deduplicate results
    unique_docs = {}
    for doc in all_docs:
        # Deduplicate by page_content
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = doc
            
    merged_docs = list(unique_docs.values())
    _log(f"Merged into {len(merged_docs)} unique chunk(s) before reranking.")
    
    # 4. Rerank the merged pool
    _log(f"Reranking top {settings.rerank_top_k} chunks using CrossEncoder...")
    reranker = get_reranker_model()
    
    # Prepare text pairs for the reranker: (rewritten_query, chunk_text)
    text_pairs = [(rewritten_query, doc.page_content) for doc in merged_docs]
    scores = reranker.score(text_pairs)
    
    # Sort documents by score descending
    doc_score_pairs = list(zip(merged_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top k
    top_docs = [doc for doc, score in doc_score_pairs[:settings.rerank_top_k]]
    _log(f"Selected top {len(top_docs)} chunk(s) after reranking.")
    
    # 5. Generate Answer
    _log(f"Generating answer with Groq ({settings.groq_model})...")
    # We can pass the top_docs directly to the prompt instead of using the retriever in the chain
    context = _format_docs(top_docs)
    answer = (RAG_PROMPT | get_llm() | StrOutputParser()).invoke({
        "context": context,
        "input": rewritten_query
    })

    sources = list({doc.metadata.get("source", "unknown") for doc in top_docs})
    _log(f"Sources: {sources}")

    return {"answer": answer, "sources": sources}
