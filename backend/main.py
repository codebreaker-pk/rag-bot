from __future__ import annotations

from pathlib import Path
import math
from typing import Optional, List, Dict
from io import BytesIO

from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import ChatRequest, ChatResponse
from utils import get_env, new_session
from ingest import load_and_ingest, extract_text_from_bytes
from rag import (
    search as rag_search,
    generate,
    generate_general,
    classify_intent,
    confidence_from_hits,
    collection_stats,
    add_docs,
    chunk,
)

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# -------- App --------
app = FastAPI(title="Business RAG Bot", version="1.0.0")

# -------- CORS --------
def _parse_origins(val: str):
    if not val or val.strip() == "*":
        return ["*"], False
    items = [x.strip() for x in val.split(",") if x.strip()]
    return items or ["*"], True

_raw = get_env("CORS_ORIGINS", "*")
_allow, _creds = _parse_origins(_raw)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow,
    allow_credentials=_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Memory --------
MEM: Dict[str, List[Dict[str, str]]] = {}

# -------- Health --------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- Debug: which files exist on server --------
@app.get("/debug/files")
def debug_files():
    def _ls(d: Path):
        if not d.exists():
            return {"exists": False, "count": 0, "files": []}
        files = [p.name for p in d.iterdir() if p.is_file()]
        return {"exists": True, "count": len(files), "files": files[:50]}
    return {
        "nec": _ls(DATA_DIR / "nec"),
        "wattmonk": _ls(DATA_DIR / "wattmonk"),
        "cwd": str(BASE_DIR),
    }

# -------- Ingest (crash-proof) --------
@app.post("/ingest")
def ingest():
    nec_dir = DATA_DIR / "nec"
    wm_dir  = DATA_DIR / "wattmonk"

    result = {"status": "", "nec": None, "wattmonk": None}

    any_loaded = False
    if nec_dir.exists():
        res_nec = load_and_ingest("nec", str(nec_dir))
        result["nec"] = res_nec
        any_loaded = any_loaded or (res_nec["files_indexed"] > 0)

    if wm_dir.exists():
        res_wm = load_and_ingest("wattmonk", str(wm_dir))
        result["wattmonk"] = res_wm
        any_loaded = any_loaded or (res_wm["files_indexed"] > 0)

    result["status"] = "ingested" if any_loaded else "no_data"
    if not any_loaded:
        result["hint"] = "Place files under backend/data/{nec|wattmonk} or use /upload"
    return result

# -------- Upload (direct file -> vector store) --------
@app.post("/upload")
async def upload(domain: str = Form(...), file: UploadFile = File(...)):
    domain = (domain or "").lower().strip()
    if domain not in {"nec", "wattmonk"}:
        raise HTTPException(status_code=400, detail="domain must be 'nec' or 'wattmonk'")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    text = extract_text_from_bytes(file.filename or "upload", data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted (unsupported or empty)")

    pieces = chunk(text)
    docs = [{
        "id": f"{domain}:{file.filename}:{i}",
        "title": file.filename,
        "source": f"upload/{file.filename}",
        "text": piece
    } for i, piece in enumerate(pieces)]
    add_docs(domain, docs)
    return {"status": "ok", "domain": domain, "filename": file.filename, "chunks": len(docs)}

# -------- Smalltalk helper --------
def is_smalltalk(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    short = len(t) <= 3
    greetings = ("hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening")
    qs = t.rstrip(" ?!.")
    return short or any(qs == g or qs.startswith(g) for g in greetings)

def _sim_from_dist(d) -> float | None:
    try:
        if d is None:
            return None
        d = float(d)
        return max(0.05, min(0.95, float(math.exp(-d))))
    except (TypeError, ValueError):
        return None

# -------- Chat --------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or new_session()
    history = MEM.get(sid, [])

    domain = req.domain if req.domain != "auto" else classify_intent(req.message)

    hits = []
    if domain in ["nec", "wattmonk"] and not is_smalltalk(req.message):
        hits = rag_search(req.message, k=5, domain_filter=domain)
        answer = generate(req.message, hits, history)
        conf = confidence_from_hits(hits)
    else:
        answer = generate_general(req.message, history)
        conf = 0.5

    sources = []
    for i, h in enumerate(hits):
        s = _sim_from_dist(h.get("dist"))
        if s is None:
            s = max(0.35, 0.85 - 0.15 * i)
        sources.append({"title": h["meta"]["title"], "doc_id": h["id"], "score": s})

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    MEM[sid] = history[-12:]

    return ChatResponse(answer=answer, sources=sources, confidence=conf, session_id=sid)

# -------- Stats --------
@app.get("/stats")
def stats():
    return collection_stats()

# -------- Debug search --------
@app.get("/debug/search")
def debug_search(q: str = Query(...), domain: Optional[str] = None):
    hits = rag_search(q, k=5, domain_filter=domain)
    return [{"id": h["id"], "title": h["meta"]["title"], "dist": h.get("dist")} for h in hits]
