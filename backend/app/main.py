"""FastAPI application entry point."""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import query, upload, session

app = FastAPI(
    title="Saarthi RAG API",
    description="PDF upload → parse → embed → retrieve → Groq response",
    version="0.1.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Print every request to stdout (always visible in terminal)."""
    start = time.perf_counter()
    print(f"[Saarthi] -> {request.method} {request.url.path}", flush=True)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(
        f"[Saarthi] <- {request.method} {request.url.path} | {response.status_code} | {elapsed_ms:.0f}ms",
        flush=True,
    )
    return response


print("[Saarthi] API module loaded", flush=True)

# Allow the Next.js frontend to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(session.router)


@app.get("/health")
async def health():
    """Simple endpoint for uptime checks."""
    print("[Saarthi] Health check OK", flush=True)
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup():
    print("[Saarthi] Server started - ready for upload/query", flush=True)
