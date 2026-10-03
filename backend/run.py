"""Start uvicorn with unbuffered stdout so logs appear immediately."""

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
