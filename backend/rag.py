from __future__ import annotations

from typing import List, Dict, Optional
from pathlib import Path
import math
import numpy as np
import chromadb
from tiktoken import get_encoding
from utils import get_env

# ====== Config ======
PROVIDER = get_env("PROVIDER", "gemini").lower()          # gemini | openai | groq
GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-2.5-flash")

# Embeddings: use Gemini to avoid Torch memory
EMBED_BACKEND = get_env("EMBED_BACKEND", "gemini").lower()  # gemini | sbert (sbert not recommended on Render free)
GEM_EMBED_MODEL = get_env("GEMINI_EMBED_MODEL", "text-embedding-004")  # will fallback automatically
SBERT_MODEL   = get_env("SBERT_MODEL", "all-MiniLM-L6-v2")  # unused if EMBED_BACKEND=gemini

DUMMY = get_env("DUMMY_MODE", "false").lower() == "true"

BASE_DIR   = Path(__file__).resolve().parent
VSTORE_DIR = BASE_DIR / "vectorstore"

# ====== Tokenizer / Chunking ======
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

# ====== Chroma (cosine space) ======
VSTORE_DIR.mkdir(parents=True, exist_ok=True)
chroma = chromadb.PersistentClient(path=str(VSTORE_DIR))
collection = chroma.get_or_create_collection(
    name="kb_main",
    metadata={"hnsw:space": "cosine"},
)

# ====== Embeddings ======
_sbert = None  # only if you ever switch to sbert

def _gemini_embed(texts: List[str]) -> List[List[float]]:
    import google.generativeai as genai
    api_key = get_env("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

    def _call(model_name: str, ts: List[str]) -> List[List[float]]:
        vecs = []
        for t in ts:
            # SDK returns object with .embedding OR dict with 'embedding'
            resp = genai.embed_content(model=model_name, content=t or "")
            emb = getattr(resp, "embedding", None)
            if emb is None and isinstance(resp, dict):
                emb = resp.get("embedding")
            if not emb:
                emb = [0.0] * 768  # safe fallback length
            vecs.append(list(emb))
        return vecs

    try:
        return _call(GEM_EMBED_MODEL, texts)
    except Exception:
        # broad fallback to widely available model
        try:
            return _call("embedding-gecko-001", texts)
        except Exception as e:
            raise RuntimeError(f"Gemini embeddings failed: {e}") from e

def _get_sbert():
    # Not recommended on Render free (Torch memory). Kept for local/dev only.
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer(SBERT_MODEL)
    return _sbert

def embed_texts(texts: List[str]) -> List[List[float]]:
    if DUMMY:
        return [[0.0] * 512 for _ in texts]
    if EMBED_BACKEND == "gemini":
        return _gemini_embed(texts)
    # sbert branch (avoid on Render free)
    model = _get_sbert()
    embs = model.encode(texts, normalize_embeddings=True)
    return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

# ====== Vector ops ======
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

# ====== Confidence ======
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

# ====== LLM (Gemini) ======
_gem_model = None
def _gemini_gen(prompt: str) -> str:
    global _gem_model
    if _gem_model is None:
        import google.generativeai as genai
        api_key = get_env("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        _gem_model = genai.GenerativeModel(GEMINI_MODEL)
    resp = _gem_model.generate_content(prompt)
    return (getattr(resp, "text", None) or "").strip()

def _llm(prompt: str) -> str:
    if DUMMY:
        if "Classify user intent" in prompt or "Return only the label" in prompt:
            return "general"
    if PROVIDER == "gemini":
        return _gemini_gen(prompt)
    return "Model unavailable: set PROVIDER=gemini with GEMINI_API_KEY."

# ====== Prompts ======
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
        "Rules:\n"
        "1) For NEC/Wattmonk questions, use ONLY the provided Context.\n"
        "2) If unsure, say you don't have that info and suggest next steps.\n"
        "3) Always include inline citations like [#] matching Context chunks.\n"
        "4) Be concise and clear."
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
    prompt = (
        "Classify user intent as one of exactly: nec, wattmonk, general.\n"
        "Output only the single lowercase label.\n\n"
        f"Message: {message}\nLabel:"
    )
    try:
        lab = (_llm(prompt) or "").strip().lower()
        return lab if lab in {"nec","wattmonk","general"} else "general"
    except Exception:
        return "general"
