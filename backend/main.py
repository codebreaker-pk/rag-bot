from pathlib import Path
import math

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse
from utils import get_env, new_session
from ingest import load_and_ingest
from rag import (
    search,
    generate,
    generate_general,
    classify_intent,
    confidence_from_hits,
    collection_stats,
)

# Paths: .../rag-bot/backend and .../rag-bot/backend/data
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="Business RAG Bot", version="1.0.0")

origins = [get_env("CORS_ORIGINS", "http://localhost:5173")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory chat history (swap with Redis later)
MEM = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest():
    load_and_ingest("nec", str(DATA_DIR / "nec"))
    load_and_ingest("wattmonk", str(DATA_DIR / "wattmonk"))
    return {"status": "ingested"}

def is_smalltalk(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    short = len(t) <= 3
    greetings = ("hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening")
    qs = t.rstrip(" ?!.")
    return short or any(qs == g or qs.startswith(g) for g in greetings)

def _sim_from_dist(d) -> float | None:
    """Works for both L2 and cosine distances -> similarity ~ exp(-d)."""
    try:
        if d is None:
            return None
        d = float(d)
        return max(0.05, min(0.95, float(math.exp(-d))))
    except (TypeError, ValueError):
        return None

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or new_session()
    history = MEM.get(sid, [])

    # Auto or explicit domain
    domain = req.domain if req.domain != "auto" else classify_intent(req.message)

    hits = []
    if domain in ["nec", "wattmonk"] and not is_smalltalk(req.message):
        hits = search(req.message, k=5, domain_filter=domain)
        answer = generate(req.message, hits, history)
        conf = confidence_from_hits(hits)
    else:
        # greetings / general chitchat -> normal chat (no citations)
        answer = generate_general(req.message, history)
        conf = 0.5

    # Build source chips with robust scoring (and rank fallback)
    sources = []
    for i, h in enumerate(hits):
        s = _sim_from_dist(h.get("dist"))
        if s is None:
            # rank fallback so it never shows 0%
            s = max(0.35, 0.85 - 0.15 * i)
        sources.append({
            "title": h["meta"]["title"],
            "doc_id": h["id"],
            "score": s,
        })

    # Update conversation memory
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    MEM[sid] = history[-12:]

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=conf,
        session_id=sid,
    )

@app.get("/stats")
def stats():
    return collection_stats()

@app.get("/debug/search")
def debug_search(q: str = Query(...), domain: str | None = None):
    hits = search(q, k=5, domain_filter=domain)
    return [{"id": h["id"], "title": h["meta"]["title"], "dist": h.get("dist")} for h in hits]
