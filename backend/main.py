from __future__ import annotations

from pathlib import Path
import math
from typing import Optional, List, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse
from utils import get_env, new_session
from ingest import load_and_ingest
from rag import (
    search as rag_search,
    generate,
    generate_general,
    classify_intent,
    confidence_from_hits,
    collection_stats,
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../rag-bot/backend
DATA_DIR = BASE_DIR / "data"                        # .../rag-bot/backend/data

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Business RAG Bot", version="1.0.0")

# -----------------------------
# CORS (beginner-safe)
# -----------------------------
def _parse_origins(val: str) -> tuple[list[str], bool]:
    """
    Returns (origins_list, allow_credentials_flag).
    If "*" -> credentials must be False (browser spec).
    """
    if not val or val.strip() == "*":
        return ["*"], False
    items = [x.strip() for x in val.split(",") if x.strip()]
    return items or ["*"], True

_raw_origins = get_env("CORS_ORIGINS", "http://localhost:5173")
_allow_list, _creds = _parse_origins(_raw_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_list,
    allow_credentials=_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Simple in-memory chat history
# -----------------------------
MEM: dict[str, List[Dict[str, str]]] = {}

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Ingest (reads data/nec and data/wattmonk)
# -----------------------------
@app.post("/ingest")
def ingest():
    load_and_ingest("nec", str(DATA_DIR / "nec"))
    load_and_ingest("wattmonk", str(DATA_DIR / "wattmonk"))
    return {"status": "ingested"}

# -----------------------------
# Small helper utilities
# -----------------------------
def is_smalltalk(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    short = len(t) <= 3
    greetings = ("hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening")
    qs = t.rstrip(" ?!.")
    return short or any(qs == g or qs.startswith(g) for g in greetings)

def _sim_from_dist(d) -> float | None:
    """Distance -> similarity mapping; robust for cosine/L2."""
    try:
        if d is None:
            return None
        d = float(d)
        # exp(-d) gives pleasant 0..1 curve for distances
        return max(0.05, min(0.95, float(math.exp(-d))))
    except (TypeError, ValueError):
        return None

# -----------------------------
# Chat
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # session
    sid = req.session_id or new_session()
    history = MEM.get(sid, [])

    # auto or explicit domain
    domain = req.domain if req.domain != "auto" else classify_intent(req.message)

    # retrieve & generate
    hits = []
    if domain in ["nec", "wattmonk"] and not is_smalltalk(req.message):
        hits = rag_search(req.message, k=5, domain_filter=domain)
        answer = generate(req.message, hits, history)
        conf = confidence_from_hits(hits)
    else:
        # smalltalk / general Qs
        answer = generate_general(req.message, history)
        conf = 0.5

    # pack sources (top-3 scored)
    sources = []
    for i, h in enumerate(hits):
        s = _sim_from_dist(h.get("dist"))
        if s is None:
            # fallback descending
            s = max(0.35, 0.85 - 0.15 * i)
        sources.append({
            "title": h["meta"]["title"],
            "doc_id": h["id"],
            "score": s,
        })

    # update memory
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    MEM[sid] = history[-12:]  # keep last 12 messages

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=conf,
        session_id=sid,
    )

# -----------------------------
# Stats
# -----------------------------
@app.get("/stats")
def stats():
    return collection_stats()

# -----------------------------
# Debug: search top-5
# -----------------------------
@app.get("/debug/search")
def debug_search(q: str = Query(...), domain: Optional[str] = None):
    hits = rag_search(q, k=5, domain_filter=domain)
    return [{"id": h["id"], "title": h["meta"]["title"], "dist": h.get("dist")} for h in hits]
