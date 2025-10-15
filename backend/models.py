from typing import Optional, List
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    domain: str = "auto"  # "auto" | "nec" | "wattmonk" | "general"

class Source(BaseModel):
    title: str
    doc_id: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    session_id: str
