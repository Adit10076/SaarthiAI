"""BGE-Reranker-v2-M3 via Hugging Face Inference API (no local model download)."""

from functools import lru_cache

import httpx
from langchain_core.cross_encoders import BaseCrossEncoder

from app.config import settings

HF_RERANK_URL = (
    f"https://router.huggingface.co/hf-inference/models/{settings.reranker_model}"
)


class HuggingFaceReranker(BaseCrossEncoder):
    """Score query–passage pairs through HF's hosted reranker."""

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        if not text_pairs:
            return []

        payload = {
            "inputs": [
                {"text": query, "text_pair": passage} for query, passage in text_pairs
            ]
        }
        response = httpx.post(
            HF_RERANK_URL,
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()

        data = response.json()
        # Batch response: [[{label, score}, ...]]
        items = data[0] if data and isinstance(data[0], list) else data
        return [float(item[0]["score"] if isinstance(item, list) else item["score"]) for item in items]


@lru_cache
def get_reranker_model() -> HuggingFaceReranker:
    """Return a cached HF Inference API reranker client."""
    return HuggingFaceReranker()
