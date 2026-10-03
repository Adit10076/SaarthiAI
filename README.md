<div align="center">

# 🧭 Saarthi AI (सारथी)
### Enterprise-Grade Advanced Retrieval-Augmented Generation (RAG) System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud_Hybrid-DC2626?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-F05A28?style=for-the-badge&logo=fastly&logoColor=white)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>A high-precision, low-latency document intelligence platform engineered with Hybrid Retrieval (Dense BGE-M3 + Sparse BM25), Query Expansion, HyDE Synthesis, Cross-Encoder Reranking, and Cookie-Isolated Multi-Tenancy.</strong>
</p>

[Key Features](#-key-features) •
[Architecture](#-system-architecture) •
[Why Saarthi vs. Naive RAG](#-why-saarthi-ai-the-engineering-advantage) •
[Tech Stack](#-technical-stack) •
[Quickstart](#-getting-started) •
[API Reference](#-api-reference) •
[Multi-Tenancy & Security](#-security--session-isolation)

---

</div>

## 📌 Overview

**Saarthi AI** (*named after the Sanskrit word for "guide" or "charioteer"*) is a production-oriented RAG system designed to overcome the classic pitfalls of first-generation vector search: semantic mismatch, keyword blindness, context dilution, and multi-user data leakage.

Instead of relying on single-vector cosine similarity, Saarthi AI implements a **multi-stage retrieval and refinement pipeline**:
1. **LlamaParse Document Extraction:** Converts complex, multi-column enterprise PDFs, tables, and headers into semantically rich Markdown.
2. **Native Hybrid Search:** Combines dense semantic vector search (`BAAI/bge-m3`, 1024 dimensions) with sparse lexical BM25 token matching via **Qdrant Cloud**.
3. **Advanced Query Transformation Engine:** Simultaneously runs **Query Rewriting**, **Multi-Query Expansion (4 variations)**, and **HyDE (Hypothetical Document Embeddings)** to bridge query-context semantic distance.
4. **Cross-Encoder Neural Reranking:** Re-scores and prunes candidate pools using `BAAI/bge-reranker-v2-m3` down to the highest-confidence chunks, eliminating needle-in-a-haystack noise.
5. **Ultra-Low Latency Inference:** Streams synthesis through **Llama 3.3 70B Versatile** on **Groq's LPU™ silicon**, delivering grounded answers with verbatim source attribution in sub-second response times.
6. **Zero-Trust Multi-Tenancy:** Enforces session isolation at the vector database index layer using cryptographically random `HttpOnly` cookie tokens.

---

## ⚡ Key Features

- **🚀 Hybrid Dense + Sparse Retrieval:** Solves the semantic-lexical tradeoff. Dense vectors capture abstract concepts; BM25 ensures part numbers, domain acronyms, and names are never missed.
- **🧠 Query Transformation Pipeline (LCEL):**
  - **Query Canonicalization:** Cleans up telegraphic or fragmented queries.
  - **Multi-Query Expansion:** Evaluates 4 parallel syntactic reformulations to capture different perspectives.
  - **HyDE (Hypothetical Document Embeddings):** Hallucinates a plausible domain answer to match against passage embeddings rather than raw questions.
- **🎯 2-Stage Cross-Encoder Reranking:** Fetches top-$k=20$ candidates and uses a deep cross-encoder to compute query-passage interaction scores, selecting top-$5$ chunks to minimize LLM context cost and hallucination.
- **🔒 True Multi-Tenant Isolation:** Session IDs are managed via secure `HttpOnly` cookies and indexed natively as Qdrant payload filters (`metadata.session_id`), preventing cross-user data exposure.
- **⚡ Serverless & Lightweight:** Leverages Groq Cloud and Hugging Face Hosted Inference APIs. No local GPU required; zero heavy model weights bundled in deployment containers.
- **💻 Modern Full-Stack Architecture:** FastAPI backend with structured logging and CORS middleware, paired with an elegant, responsive Next.js 14 / TypeScript interface.

---

## 🏗 System Architecture

The following diagram illustrates the lifecycle of document ingestion and query execution through the Saarthi AI pipeline:

```mermaid
flowchart TB
    subgraph INGESTION["1. Document Ingestion Pipeline"]
        PDF["Uploaded PDF Document"] --> LP["LlamaParse Engine<br/>(Markdown Extraction)"]
        LP --> Splitter["Recursive Character Splitter<br/>(Chunk: 1000 | Overlap: 200)"]
        Splitter --> MetaTag["Metadata Enrichment<br/>(Session ID, Document ID, Source)"]
        MetaTag --> DenseEnc["Dense Encoder<br/>(BAAI/bge-m3 via HF API)"]
        MetaTag --> SparseEnc["Sparse Encoder<br/>(BM25 via FastEmbed)"]
        DenseEnc --> QdrantUpsert["Qdrant Cloud Vector Store<br/>(Hybrid Index + Session Payload)"]
        SparseEnc --> QdrantUpsert
    end

    subgraph QUERY["2. Advanced Query & Transformation Engine"]
        UserQuery["User Natural Language Query"] --> Rewriter["Query Rewriter<br/>(Canonicalization)"]
        Rewriter --> Parallel["RunnableParallel Execution"]
        Parallel --> MQ["Multi-Query Expansion<br/>(4 Diverse Formulations)"]
        Parallel --> HyDE["HyDE Generator<br/>(Hypothetical Answer Passage)"]
    end

    subgraph RETRIEVAL["3. Hybrid Search & Neural Reranking"]
        MQ --> HybridSearch["Qdrant Native Hybrid Search<br/>(Filter: session_id | k=20)"]
        HyDE --> HybridSearch
        Rewriter --> HybridSearch
        HybridSearch --> Dedupe["Deduplication & Candidate Pooling"]
        Dedupe --> Reranker["Cross-Encoder Reranker<br/>(BAAI/bge-reranker-v2-m3)"]
        Reranker --> TopK["Top-5 Most Relevant Chunks"]
    end

    subgraph GENERATION["4. Synthesis & Response"]
        TopK --> PromptContext["Grounded Context Formatter"]
        PromptContext --> GroqLLM["Groq LPU Inference<br/>(Llama 3.3 70B Versatile)"]
        GroqLLM --> Output["Grounded Answer + Source Citations"]
    end

    UserQuery -. Cookie Session .- QdrantUpsert
```

---

## 🔬 Why Saarthi AI? (The Engineering Advantage)

Standard tutorials promote **"Naive RAG"** (chunk text $\to$ generate single vector $\to$ fetch nearest neighbors $\to$ call OpenAI). In production enterprise environments, Naive RAG breaks down:

| Vector Search Dimension | Naive RAG (Baseline) | Saarthi AI Architecture |
| :--- | :--- | :--- |
| **Parsing Quality** | Raw text extraction; drops tables, formatting, headers. | **LlamaParse Markdown Extraction**: preserves tabular structure and markdown semantics. |
| **Search Paradigm** | Dense vector cosine similarity only. Misses precise keywords and acronyms. | **Native Hybrid Search**: Fuses dense `bge-m3` vectors with sparse `BM25` for lexical + semantic recall. |
| **Query Representation** | Matches raw question against answer passages (asymmetric semantic mismatch). | **HyDE + Multi-Query**: Generates hypothetical answer embeddings and 4 diverse semantic queries. |
| **Context Quality** | Top-K bi-encoder candidates sent directly to LLM (high context dilution / noise). | **Cross-Encoder Neural Reranking**: Scores full query-document attention matrices to retain only Top-5 chunks. |
| **Generation Speed** | Standard cloud GPU inference (30–60 tokens/sec). | **Groq LPU Acceleration**: 300+ tokens/sec on Llama 3.3 70B with near-instant time-to-first-token. |
| **Data Privacy** | Shared index without isolation; risk of cross-tenant prompt injection or leaks. | **HttpOnly Session Cookies + Qdrant Payload Indexing**: Hardware-isolated tenant boundaries. |
| **Infrastructure Cost** | Requires high-VRAM dedicated GPU instances (A10G / A100). | **Serverless Inference**: Hosted HF API + Groq + Qdrant Cloud. Zero dedicated GPU idle cost. |

---

## 🛠 Technical Stack

### Backend
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous ASGI, dependency injection, type validation)
- **Orchestration:** [LangChain](https://www.langchain.com/) Core & LCEL (`RunnableParallel`, `ChatPromptTemplate`, `StrOutputParser`)
- **Document Parsing:** [LlamaParse](https://cloud.llamaindex.ai/) (Cloud multimodal markdown parser)
- **Vector Database:** [Qdrant Cloud](https://qdrant.tech/) (Native hybrid dense + sparse search, payload schema indexing)
- **Embeddings:** `BAAI/bge-m3` via Hugging Face Hosted Inference API + `Qdrant/bm25` via FastEmbed
- **Reranking:** `BAAI/bge-reranker-v2-m3` via Hugging Face Inference API (`BaseCrossEncoder`)
- **LLM Engine:** `llama-3.3-70b-versatile` running on [Groq LPU](https://groq.com/)
- **Configuration:** [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) with automatic `.env` resolution

### Frontend
- **Framework:** [Next.js 14](https://nextjs.org/) (App Router, Server & Client Components)
- **Language:** [TypeScript](https://www.typescriptlang.org/) (Strict typing, interface contracts)
- **Styling:** Vanilla CSS design system with CSS custom properties (dark-mode glassmorphism, responsive grid)
- **State Management:** React Hooks (`useState`, `useEffect`) with credentialed `fetch` API

---

## 📂 Repository Structure

```text
Saarthi AI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── dependencies.py          # HttpOnly cookie session validator
│   │   │   ├── query.py                 # POST /api/query route handler
│   │   │   ├── session.py               # GET /api/session cookie generator
│   │   │   ├── upload.py                # POST /api/upload PDF processing pipeline
│   │   │   └── services/
│   │   │       ├── advanced_retrieval.py# Query Rewriter, Multi-Query, and HyDE chains
│   │   │       ├── embeddings.py        # Dense (BGE-M3) & Sparse (BM25) embedders
│   │   │       ├── llm.py               # Cached ChatGroq client provider
│   │   │       ├── parser.py            # LlamaParse PDF-to-Markdown loader
│   │   │       ├── rag.py               # End-to-end RAG orchestrator & context joiner
│   │   │       ├── reranker.py          # HuggingFace Cross-Encoder client
│   │   │       ├── retrieval.py         # Base ensemble retriever with Qdrant filters
│   │   │       └── vector_store.py      # Qdrant client, chunking, and upsert logic
│   │   ├── config.py                    # Central Pydantic BaseSettings & cache dirs
│   │   └── main.py                      # FastAPI app entry point, CORS, and request logger
│   ├── tests/
│   │   └── test_session_isolation.py    # Pytest unit & integration tests
│   ├── .env.example                     # Environment template for backend
│   ├── requirements.txt                 # Pinned Python dependencies
│   └── run.py                           # Unbuffered development server launcher
│
├── frontend/
│   ├── app/
│   │   ├── globals.css                  # Modern UI tokens, responsive layouts, badges
│   │   ├── layout.tsx                   # Root HTML shell & metadata
│   │   └── page.tsx                     # Main dashboard with pipeline stages
│   ├── components/
│   │   ├── PdfUpload.tsx                # File dropzone & real-time ingestion status
│   │   └── QueryChat.tsx                # Conversational search bar & verified sources
│   ├── lib/
│   │   └── api.ts                       # Typed API client with credentialed sessions
│   ├── .env.local.example               # Frontend environment template
│   ├── package.json                     # Next.js & React dependencies
│   └── tsconfig.json                    # Strict TypeScript configuration
│
└── README.md                            # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python:** `3.10+` (3.11 recommended)
- **Node.js:** `18.x+` or `20.x+`
- **Package Managers:** `pip` and `npm`

### 2. Obtain API Keys
You will need API keys from the following services (all offer free tiers):
1. **[Groq Cloud Console](https://console.groq.com/):** For ultra-fast Llama 3.3 LLM inference.
2. **[LlamaIndex Cloud](https://cloud.llamaindex.ai/):** For LlamaParse document extraction.
3. **[Hugging Face](https://huggingface.co/settings/tokens):** Fine-grained token with *Make calls to Inference Providers* enabled.
4. **[Qdrant Cloud](https://cloud.qdrant.io/):** Managed Qdrant cluster URL and API key.

---

### 3. Backend Setup

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate a Python virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

Configure your `.env` file:
```ini
# Groq LLM
GROQ_API_KEY=gsk_your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# LlamaParse document parser
LLAMA_CLOUD_API_KEY=llx_your_llama_cloud_api_key

# Hugging Face Inference API
HF_TOKEN=hf_your_huggingface_token
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Qdrant Cloud
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=saarthi_docs_hybrid

# Hybrid retrieval tuning
RETRIEVAL_K=20
RERANK_TOP_K=5
```

Start the FastAPI backend:
```bash
python run.py
```
> The API will be available at `http://127.0.0.1:8000` with interactive Swagger docs at `http://127.0.0.1:8000/docs`.

---

### 4. Frontend Setup

```bash
# Open a new terminal and navigate to frontend directory
cd frontend

# 1. Install dependencies
npm install

# 2. Configure environment variables
cp .env.local.example .env.local
```

Ensure `frontend/.env.local` contains:
```ini
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the Next.js development server:
```bash
npm run dev
```
> Open [http://localhost:3000](http://localhost:3000) in your browser to interact with the application.

---

## 📡 API Reference

### `GET /api/session`
Initializes a new session by returning an `HttpOnly`, `SameSite=Lax` cookie (`saarthi_session_id`).
- **Response:**
  ```json
  { "message": "Session initialized" }
  ```

### `POST /api/upload`
Uploads a PDF, parses structured markdown via LlamaParse, splits text into chunks, and stores dense + sparse embeddings in Qdrant indexed under the user's session ID.
- **Request:** `multipart/form-data` with `file: <binary_pdf>`
- **Response:**
  ```json
  {
    "message": "PDF processed and stored successfully.",
    "filename": "quarterly_earnings_report.pdf",
    "chunks_stored": 42
  }
  ```

### `POST /api/query`
Executes query rewriting, multi-query expansion, HyDE generation, hybrid retrieval filtered by session, cross-encoder reranking, and Groq answer synthesis.
- **Request:**
  ```json
  {
    "question": "What were the primary revenue drivers in Q3 according to the report?"
  }
  ```
- **Response:**
  ```json
  {
    "answer": "According to the report, the primary revenue drivers in Q3 were enterprise SaaS expansions (up 28% YoY) and increased cloud infrastructure utilization across European regions.",
    "sources": [
      "quarterly_earnings_report.pdf"
    ]
  }
  ```

### `GET /health`
Liveness and health-check endpoint.
- **Response:** `{ "status": "ok" }`

---

## 🔒 Security & Session Isolation

Multi-tenant RAG systems must prevent cross-session context bleeding. Saarthi AI implements end-to-end tenant separation:

```text
[Browser Client]
       │
       ▼  (Sets HttpOnly Cookie: saarthi_session_id=uuid4)
[FastAPI Middleware & Dependencies]
       │
       ▼  (Extracts & validates session_id)
[Qdrant Search Filter]
       │
       └──> REST Filter: must [ FieldCondition("metadata.session_id" == session_id) ]
```

1. **HttpOnly Cookie Standard:** Prevents JavaScript in the browser from reading or manipulating the session identifier, mitigating XSS attacks.
2. **Qdrant Payload Filtering:** Retrieval executes on the Qdrant cluster using native index filtering on `metadata.session_id`. Queries from Session A will **never** match or retrieve documents uploaded by Session B.
3. **Automated Verification:** Verified by continuous integration tests in [`tests/test_session_isolation.py`](file:///d:/Saarthi%20AI/backend/tests/test_session_isolation.py).

---

## 🧪 Testing & Quality Assurance

Saarthi AI includes automated test suites covering API session isolation, cookie handling, and dependency mocking:

```bash
cd backend
python -m pytest
```

Example test output:
```text
tests/test_session_isolation.py::test_missing_session_id PASSED          [ 50% ]
tests/test_session_isolation.py::test_session_isolation PASSED           [ 100% ]
============================== 2 passed in 0.45s ==============================
```

---

## 🗺 Production Roadmap & Future Enhancements

- [x] Native Hybrid Search (BM25 + BGE-M3 Dense)
- [x] Multi-Query Expansion & HyDE Synthesis
- [x] Cross-Encoder Reranking (`bge-reranker-v2-m3`)
- [x] Multi-Tenant Session Isolation via Cookies
- [ ] **Streaming Token Generation (SSE):** Add Server-Sent Events to stream Groq response tokens word-by-word.
- [ ] **Conversational Memory:** Add Redis-backed sliding window chat history for multi-turn contextual dialog.
- [ ] **Multi-Modal Document Visuals:** Retain chart and diagram images alongside extracted tables using vision LLMs.
- [ ] **LangSmith Observability:** Integrated tracing for retrieval latency, token counts, and hallucination evaluation.

---

## 👨‍💻 Author

**Adit**
- GitHub: [@yourusername](https://github.com)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com)

---

<div align="center">
  <sub>Engineered with precision for advanced document intelligence. If you find this architecture insightful, please star ⭐ the repository!</sub>
</div>
