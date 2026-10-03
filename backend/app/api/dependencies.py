from fastapi import Cookie, HTTPException

def get_session_id(saarthi_session_id: str | None = Cookie(None)) -> str:
    """
    Extract and validate the session ID from the HttpOnly cookie.
    Returns HTTP 400 if missing.
    """
    if not saarthi_session_id:
        raise HTTPException(
            status_code=400, 
            detail="Session cookie is missing. Please initialize a session first."
        )
    return saarthi_session_id
