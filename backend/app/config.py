"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to backend/ so uvicorn works from any cwd
BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """Central config for all external service credentials and defaults."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
    )

    # Groq LLM
    groq_api_key: str

    # LlamaParse
    llama_cloud_api_key: str

    # Hugging Face (BGE-M3 embeddings via Inference API)
    hf_token: str

    # Qdrant Cloud
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "saarthi_docs_hybrid"

    # LLM model name on Groq
    groq_model: str = "llama-3.3-70b-versatile"

    # BGE-M3 embedding model
    embedding_model: str = "BAAI/bge-m3"

    # BGE reranker (FlagEmbedding, runs locally)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_use_fp16: bool = False  # set True if running on GPU

    # Hybrid retrieval: fetch top-k candidates, rerank down to rerank_top_k
    retrieval_k: int = 20
    rerank_top_k: int = 5

    # Local HuggingFace model cache on project drive (D:)
    model_cache_dir: Path = BACKEND_DIR / ".cache" / "huggingface"


settings = Settings()


def configure_hf_cache() -> None:
    """Force all HuggingFace downloads to the project drive, not C:."""
    cache_root = settings.model_cache_dir
    hub_cache = cache_root / "hub"
    cache_root.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)


configure_hf_cache()
