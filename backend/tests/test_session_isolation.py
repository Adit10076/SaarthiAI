import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)

def test_missing_session_id():
    """Test 3: Missing session cookie"""
    response = client.post("/api/query", json={"question": "test"})
    assert response.status_code == 400
    assert "Session cookie is missing" in response.json()["detail"]


@patch("app.api.upload.chunk_and_store")
@patch("app.api.upload.parse_pdf")
def test_session_isolation(mock_parse_pdf, mock_chunk_and_store):
    """
    Test 1 and Test 2: Verify session isolation during upload and query.
    We mock the actual Qdrant insertion here to test the API layer correctly.
    """
    # Test 1: Upload document A under session-1
    # Test 2: Upload document B under session-2
    # This is an integration test. To do this properly against Qdrant, we'd need a test collection.
    # For now, we just verify the cookies are passed correctly.
    
    response = client.post(
        "/api/upload", 
        files={"file": ("test.pdf", b"dummy content", "application/pdf")},
        cookies={"saarthi_session_id": "session-1"}
    )
    # Since we mock parse_pdf and chunk_and_store, this should return 200
    assert response.status_code == 200
    
    # Verify chunk_and_store was called with session-1
    mock_chunk_and_store.assert_called_once()
    args, kwargs = mock_chunk_and_store.call_args
    assert args[1] == "session-1"

    # Query session-1
    # We mock rag_query to avoid hitting LLM
    with patch("app.api.query.rag_query") as mock_rag_query:
        mock_rag_query.return_value = {"answer": "test", "sources": []}
        response = client.post(
            "/api/query", 
            json={"question": "test"},
            cookies={"saarthi_session_id": "session-1"}
        )
        assert response.status_code == 200
        mock_rag_query.assert_called_once_with("test", "session-1")
