"""PDF parsing via LlamaParse."""

from llama_parse import LlamaParse
from langchain_core.documents import Document

from app.config import settings


def parse_pdf(file_path: str) -> list[Document]:
    """
    Parse a PDF file into LangChain Document objects.

    LlamaParse returns markdown-formatted text, which preserves
    structure better than raw PDF extraction.
    """
    parser = LlamaParse(
        api_key=settings.llama_cloud_api_key,
        result_type="markdown",
        verbose=False,
    )

    # LlamaParse returns LlamaIndex documents; convert to LangChain format
    llama_docs = parser.load_data(file_path)

    return [
        Document(
            page_content=doc.text,
            metadata={"source": file_path, **doc.metadata},
        )
        for doc in llama_docs
    ]
