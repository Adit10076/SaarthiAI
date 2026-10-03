"""BGE-M3 embeddings via Hugging Face Inference API (no local model download)."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import FastEmbedSparse

from app.config import settings


@lru_cache
def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """
    Return a cached BGE-M3 embedding client.

    Calls Hugging Face's hosted inference API — no torch or local weights needed.
    """
    return HuggingFaceEndpointEmbeddings(
        model=settings.embedding_model,
        task="feature-extraction",
        provider="hf-inference",
        huggingfacehub_api_token=settings.hf_token,
    )


@lru_cache
def get_sparse_embeddings() -> FastEmbedSparse:
    """
    Return a cached FastEmbedSparse client for BM25.
    Runs locally, but it's extremely lightweight.
    """
    return FastEmbedSparse(
        model_name="Qdrant/bm25",
        cache_dir=str(settings.model_cache_dir)
    )
