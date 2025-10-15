from __future__ import annotations

from typing import List, Dict, Optional
from pathlib import Path
import math
import numpy as np
import chromadb
from tiktoken import get_encoding
from utils import get_env

# ================= Config =================
PROVIDER = get_env("PROVIDER", "gemini").lower()
GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-2.5-flash")

# Embeddings: keep it light on Render free tier
EMBED_BACKEND = get_env("EMBED_BACKEND", "gemini").lower()  # gemini | sbert (avoid sbert on 512MB)
GEMINI_EMBED_MODEL = get_env("GEMINI_EMBED_MODEL", "text-embedding-004")
DUMMY = get_env("DUMMY_MODE", "false").lower() == "true"

BASE_DIR = Path(__file__).resolve().parent
VSTORE_DIR = BASE_DIR / "vectorstore"

# ================= Chunking =================
try:
    enc = get_encoding("cl100k_base")
except Exception:
    enc = None

def chunk(text: str, max_tokens: int = 400, overlap: int = 80) -> List[str]:
    text = text or ""
    if not text.strip():
        return []
    if enc is None:
        max_chars = max_tokens * 4
        step = max(1, max_chars - overlap * 4)
        return [text[i:i+max_chars] for i in range(0, len(text), step)]
    ids = enc.encode(text)
    out, step = [], max(1, max_tokens - overlap)
    for i in range(0, len(ids), step):
        piece = enc.decode(ids[i:i+max_tokens])
        if piece.strip():
            out.append(piece)
    return out

# ================= Chroma (cosine) =================
VSTORE_DIR.mkdir(parents=True, exist_ok=True)
chroma = chromadb.PersistentClient(path=str(VSTORE_DIR))
collection = chroma.get_or_create_collection(
    name="kb_main",
    metadata={"hnsw:space": "cosine"},
)

# ================= Embeddings =================
def _gemini_embed(texts: List[str]) -> List[List[float]]:
    import google.generativeai as genai
    api_key = get_env("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

    def _call(model_name: str, ts: List[str]) -> List[List[float]]:
        vecs = []
        for t in ts:
            resp = genai.embed_content(model=model_name, content=t or "")
            emb = getattr(resp, "embedding", None)
            if emb is None and isinstance(resp, dict):
                emb = resp.get("embedding")
            if not emb:
                emb = [0.0] * 768
            vecs.append(list(emb))
        return vecs

    try:
        return _call(GEMINI_EMBED_MODEL, texts)
    except Exception:
        return _call("embedding-gecko-001", texts)

def embed_texts(texts: List[str]) -> List[List[float]]:
    if DUMMY:
        return [[0.0] * 256 for _ in texts]
    # Only Gemini path enabled for Render
    return _gemini_embed(texts)

# ================= Vector ops =================
def add_docs(domain: str, docs: List[Dict]):
    if not docs:
        return
    texts = [d.get("text", "") for d in docs]
    embs  = embed_texts(texts)
    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embs,
        metadatas=[{"domain": domain, "title": d["title"], "source": d["source"]} for d in docs],
        documents=texts,
    )

def search(query: str, k: int = 5, domain_filter: Optional[str] = None):
    if not query.strip():
        return []
    qemb  = embed_texts([query])[0]
    where = {"domain": domain_filter} if domain_filter else {}
    res = collection.query(
        query_embeddings=[qemb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    n = len(res.get("ids", [[]])[0])
    for i in range(n):
        hits.append({
            "id":   res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "dist": res.get("distances", [[None]])[0][i],
        })
    return hits

def collection_stats():
    try: nec = collection.count(where={"domain": "nec"})
    except Exception: nec = 0
    try: wm  = collection.count(where={"domain": "wattmonk"})
    except Exception: wm = 0
    return {"total": nec + wm, "nec_count": nec, "wattmonk_count": wm, "nec_samples": {}, "wattmonk_samples": {}}

def confidence_from_hits(hits: List[Dict]):
    if not hits:
        return 0.2
    ds = []
    for h in hits:
        try:
            ds.append(float(h.get("dist")))
        except (TypeError, ValueError):
            pass
    if not ds:
        return 0.4
    sim = float(math.exp(-float(np.mean(ds))))
    return max(0.3, min(0.95, sim))

# ================= LLM (Gemini) =================
_gem_model = None
def _gemini_gen(prompt: str) -> str:
    global _gem_model
    import google.generativeai as genai
    api_key = get_env("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    if _gem_model is None:
        _gem_model = genai.GenerativeModel(GEMINI_MODEL)
    resp = _gem_model.generate_content(prompt)
    return (getattr(resp, "text", None) or "").strip()

def _llm(prompt: str) -> str:
    if DUMMY:
        # simple canned response for tests
        return "Hello! How can I help you today?"
    return _gemini_gen(prompt)

# ================= Prompting =================
def _format_history(history: List[Dict]) -> str:
    return "\n".join(f'{h.get("role","user").capitalize()}: {h.get("content","")}' for h in history[-6:])

def build_prompt(message: str, ctx: List[Dict], history: List[Dict]):
    cites, ctx_text = [], ""
    for i, c in enumerate(ctx, 1):
        m = c.get("meta", {})
        cites.append(f"[{i}] {m.get('title','')} — {m.get('source','')}")
        ctx_text += f"\n[{i}] {c.get('text','')}\n"
    sys = (
        "You are a precise business assistant.\n"
        "Use ONLY the provided Context for domain questions (NEC/Wattmonk).\n"
        "If unsure, say you don't have that info and suggest next steps.\n"
        "Always include inline citations like [#] matching Context.\n"
        "Be concise and clear."
    )
    hist_txt = _format_history(history)
    user = (
        f"{sys}\n\nQuestion: {message}\n\n"
        f"Context:{ctx_text if ctx_text.strip() else ' (no context)'}\n\n"
        f"Recent history:\n{hist_txt}\n\nAnswer:"
    )
    return user, cites

def generate(message: str, ctx: List[Dict], history: List[Dict]) -> str:
    prompt, _ = build_prompt(message, ctx, history)
    out = _llm(prompt)
    return out or "I don't have that info from the provided context."

def generate_general(message: str, history: List[Dict]) -> str:
    sys = "You are a friendly, helpful assistant. Be concise."
    prompt = f"{sys}\n\nUser: {message}\nAssistant:"
    return _llm(prompt)

def classify_intent(message: str) -> str:
    m = (message or "").lower()
    if "wattmonk" in m:
        return "wattmonk"
    if any(k in m for k in ["nec", "grounded", "conductor", "article", "national electrical code"]):
        return "nec"
    # fallback classification via LLM (optional)
    return "general"
