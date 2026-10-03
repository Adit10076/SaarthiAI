"""Advanced RAG retrieval techniques: Query Rewriting, Multi-Query, and HyDE."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from app.api.services.llm import get_llm

def _log(msg: str) -> None:
    print(f"[Saarthi] {msg}", flush=True)

# 1. Query Rewriting
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant. Rewrite the following user query to be a fully formed, "
    "detailed, and clear question. If it's just keywords, turn it into a question. "
    "Do not answer the question, just rewrite it. Output ONLY the rewritten question without any prefixes.\n\n"
    "Original Query: {question}"
)

rewrite_chain = REWRITE_PROMPT | get_llm() | StrOutputParser()

# 2. Query Expansion / Multi-Query
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant. Generate 4 variations of the following query to help "
    "retrieve relevant documents from a vector database. Use synonyms and different "
    "phrasings. Return ONLY the variations, separated by newlines, without any prefixes or numbering.\n\n"
    "Original Query: {question}"
)

def parse_variations(text: str) -> list[str]:
    lines = text.strip().split("\n")
    return [line.strip("- *1234567890.") for line in lines if line.strip()]

multi_query_chain = MULTI_QUERY_PROMPT | get_llm() | StrOutputParser() | parse_variations

# 3. HyDE (Hypothetical Document Embeddings)
HYDE_PROMPT = ChatPromptTemplate.from_template(
    "You are an expert. Write a hypothetical, brief, and factual answer to the "
    "following question. Do not include any filler text, just the hypothetical answer.\n\n"
    "Question: {question}\n"
    "Hypothetical Answer:"
)

hyde_chain = HYDE_PROMPT | get_llm() | StrOutputParser()

def transform_query(question: str) -> dict:
    """
    Apply advanced retrieval transformations to the original question.
    Returns a dict with 'rewritten', 'variations', and 'hyde_doc'.
    """
    _log(f"Applying Query Rewriting to: '{question}'")
    rewritten = rewrite_chain.invoke({"question": question}).strip()
    _log(f"Rewritten Query: '{rewritten}'")

    _log("Applying Query Expansion (Multi-Query) and HyDE in parallel...")
    
    parallel_chain = RunnableParallel(
        variations=multi_query_chain,
        hyde_doc=hyde_chain
    )
    
    results = parallel_chain.invoke({"question": rewritten})
    
    _log(f"Generated {len(results['variations'])} variations.")
    _log(f"Generated HyDE hypothetical document (length: {len(results['hyde_doc'])} chars).")
    
    return {
        "rewritten": rewritten,
        "variations": results["variations"],
        "hyde_doc": results["hyde_doc"]
    }
