"""Qdrant Cloud vector store operations."""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.api.services.embeddings import get_embeddings, get_sparse_embeddings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant Cloud client."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def get_vector_store() -> QdrantVectorStore:
    """Return a QdrantVectorStore wired to the configured collection with Hybrid Search."""
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )


def chunk_and_store(documents: list[Document], session_id: str) -> int:
    """
    Split documents into chunks, embed them, and ups into Qdrant.

    Returns the number of chunks stored.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Saarthi] Split into {len(chunks)} chunk(s)", flush=True)

    document_id = str(uuid.uuid4())
    for doc in chunks:
        doc.metadata["session_id"] = session_id
        doc.metadata["document_id"] = document_id
        doc.metadata["file_name"] = doc.metadata.get("source", "unknown")
        doc.metadata["chunk_text"] = doc.page_content

    print(f"[Saarthi] Embedding + upserting to Qdrant ({settings.qdrant_collection}) with Hybrid Search...", flush=True)
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    
    # Ensure a payload index exists for session_id so we can filter by it
    client = get_qdrant_client()
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="metadata.session_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    print(f"[Saarthi] Qdrant upsert complete", flush=True)

    return len(chunks)
