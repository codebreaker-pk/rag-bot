from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Domain = Literal["nec", "wattmonk", "general", "auto"]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    domain: Domain = "auto"

class Source(BaseModel):
    title: str
    doc_id: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    confidence: float = 0.0
    session_id: str
