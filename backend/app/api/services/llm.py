"""Shared LLM instance."""

from functools import lru_cache
from langchain_groq import ChatGroq
from app.config import settings

@lru_cache
def get_llm() -> ChatGroq:
    """Return a cached Groq chat model."""
    return ChatGroq(
        model=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=0,
    )
