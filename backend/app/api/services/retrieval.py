"""Hybrid retrieval: Qdrant Native Hybrid Search, then cross-encoder reranking."""

from functools import lru_cache

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from app.config import settings
from app.api.services.reranker import get_reranker_model
from app.api.services.vector_store import get_vector_store


from qdrant_client.http import models as rest

@lru_cache(maxsize=128)
def get_base_ensemble_retriever(session_id: str):
    """
    Return the base Qdrant retriever (Dense + Sparse Hybrid) without reranking,
    filtered by session_id.
    """
    qdrant_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="metadata.session_id",
                match=rest.MatchValue(value=session_id)
            )
        ]
    )
    return get_vector_store().as_retriever(
        search_kwargs={
            "k": settings.retrieval_k,
            "filter": qdrant_filter
        }
    )


@lru_cache(maxsize=128)
def get_hybrid_retriever(session_id: str):
    """
    Build the full retrieval pipeline:

    1. Qdrant Native Hybrid Search (Dense + Sparse) filtered by session_id
    2. CrossEncoderReranker (BGE-Reranker-v2-M3) → top 5
    """
    print(
        f"[Saarthi] Building hybrid retriever "
        f"(k={settings.retrieval_k}, rerank_top={settings.rerank_top_k})",
        flush=True,
    )
    base_retriever = get_base_ensemble_retriever(session_id)

    reranker = CrossEncoderReranker(
        model=get_reranker_model(),
        top_n=settings.rerank_top_k,
    )

    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker,
    )
