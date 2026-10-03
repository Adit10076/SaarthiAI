import uuid
from fastapi import APIRouter, Response, Cookie

router = APIRouter(prefix="/api", tags=["session"])

@router.get("/session")
async def create_session(response: Response, saarthi_session_id: str | None = Cookie(None)):
    """Generate and set a new unique session ID in an HttpOnly cookie."""
    if saarthi_session_id:
        return {"message": "Session already exists"}
        
    session_id = str(uuid.uuid4())
    response.set_cookie(
        key="saarthi_session_id",
        value=session_id,
        httponly=True,
        secure=True,  # Set to True for HTTPS, but works locally in most modern browsers
        samesite="lax",
    )
    return {"message": "Session initialized"}
