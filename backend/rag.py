from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import chromadb
from tiktoken import get_encoding
from utils import get_env

# ===== Settings via .env =====
PROVIDER = get_env("PROVIDER", "gemini").lower()          # 'gemini' | 'groq' | 'openai'
EMBED_BACKEND = get_env("EMBED_BACKEND", "sbert").lower() # 'sbert' | 'openai'
DUMMY = get_env("DUMMY_MODE", "false").lower() == "true"

# ===== Vector DB (local persistent) =====
VSTORE_DIR = Path(__file__).resolve().parent / "vectorstore"
VSTORE_DIR.mkdir(parents=True, exist_ok=True)
chroma = chromadb.PersistentClient(path=str(VSTORE_DIR))
collection = chroma.get_or_create_collection(name="kb_main")  # >=3 chars

# ===== Tokenizer for chunking =====
enc = get_encoding("cl100k_base")

# ===== Embeddings =====
if EMBED_BACKEND == "openai" and not DUMMY:
    from openai import OpenAI
    _openai = OpenAI()
    EMBED_MODEL = get_env("EMBED_MODEL", "text-embedding-3-small")
    def embed_texts(texts: List[str]) -> List[List[float]]:
        out = _openai.embeddings.create(model=EMBED_MODEL, input=texts)
        return [e.embedding for e in out.data]
else:
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer(get_env("SBERT_MODEL", "all-MiniLM-L6-v2"))
    def embed_texts(texts: List[str]) -> List[List[float]]:
        return _sbert.encode(texts, normalize_embeddings=True).tolist()

# ===== Chat providers =====
def _chat_openai(messages):
    from openai import OpenAI
    cli = OpenAI()
    model = get_env("MODEL_NAME", "gpt-4o-mini")
    r = cli.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return r.choices[0].message.content

def _chat_groq(messages):
    from groq import Groq
    cli = Groq(api_key=get_env("GROQ_API_KEY"))
    model = get_env("GROQ_MODEL", "llama-3.1-8b-instant")
    r = cli.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return r.choices[0].message.content

def _pick_gemini_model(client):
    preferred = get_env("GEMINI_MODEL", "").strip()
    if preferred:
        return preferred
    # auto-detect available model
    names = [m.name.split("/")[-1] for m in client.models.list()]
    for cand in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]:
        if cand in names:
            return cand
    if names:
        return names[0]
    raise RuntimeError("No Gemini generative models available for this key.")

def _chat_gemini(messages):
    from google import genai
    client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
    model = _pick_gemini_model(client)

    # Convert OpenAI-style messages -> Gemini contents
    sys_msg = ""
    msgs = messages[:]
    if msgs and msgs[0].get("role") == "system":
        sys_msg = msgs[0].get("content", "")
        msgs = msgs[1:]

    contents = []
    if sys_msg:
        contents.append({"role": "user", "parts": [{"text": sys_msg}]})
    for m in msgs:
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

    resp = client.models.generate_content(model=model, contents=contents)
    text = getattr(resp, "text", None)
    if not text:
        cand = getattr(resp, "candidates", None)
        if cand and getattr(cand[0], "content", None):
            parts = getattr(cand[0].content, "parts", [])
            text = "".join(getattr(p, "text", "") for p in parts)
    return text or ""

def chat_complete(messages):
    if DUMMY:
        raise RuntimeError("DUMMY_MODE on")
    if PROVIDER == "gemini":
        return _chat_gemini(messages)
    if PROVIDER == "groq":
        return _chat_groq(messages)
    if PROVIDER == "openai":
        return _chat_openai(messages)
    raise RuntimeError(f"Unknown PROVIDER={PROVIDER}")

# ===== RAG helpers =====
def chunk(text: str, max_tokens=400, overlap=80):
    ids = enc.encode(text or "")
    out = []
    step = max(1, max_tokens - overlap)
    for i in range(0, len(ids), step):
        piece = enc.decode(ids[i:i + max_tokens])
        if piece.strip():
            out.append(piece)
    return out or [text]

def add_docs(domain: str, docs: List[Dict]):
    if not docs:
        return
    texts = [d["text"] for d in docs]
    embs = embed_texts(texts)
    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embs,
        metadatas=[{"domain": domain, "title": d["title"], "source": d["source"]} for d in docs],
        documents=texts,
    )

def search(query: str, k: int = 5, domain_filter: Optional[str] = None):
    if DUMMY: return []
    qemb = embed_texts([query])[0]
    where = {"domain": domain_filter} if domain_filter else {}
    res = collection.query(
        query_embeddings=[qemb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],  # <<< IMPORTANT
    )
    hits = []
    for i in range(len(res.get("ids", [[""]])[0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "dist": res.get("distances", [[None]])[0][i],  # keep None if missing
        })
    return hits


def classify_intent(message: str) -> str:
    try:
        label = chat_complete([
            {"role": "system", "content": "Classify user intent as one of: nec, wattmonk, general. Return only the label."},
            {"role": "user", "content": message},
        ]).strip().lower()
        return label if label in ["nec", "wattmonk", "general"] else "general"
    except Exception:
        m = (message or "").lower()
        if "nec" in m: return "nec"
        if "wattmonk" in m or "solar" in m: return "wattmonk"
        return "general"

def build_prompt(message: str, ctx: List[Dict], history: List[Dict]):
    ctx_text = ""
    for i, c in enumerate(ctx, 1):
        ctx_text += f"\n[{i}] {c['text']}\n"
    hist_txt = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history[-6:])
    sys = ("You are a helpful assistant. Use ONLY the provided context for domain questions. "
           "If unsure, say you don't have that info and propose next steps. Always include inline [#] cites.")
    user = f"Question: {message}\n\nContext:\n{ctx_text}\n\nHistory:\n{hist_txt}\n"
    return sys, user

def generate(message: str, ctx: List[Dict], history: List[Dict]):
    try:
        sys, user = build_prompt(message, ctx, history)
        return chat_complete([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    except Exception as e:
        if ctx:
            snippet = "\n\n".join([f"[{i+1}] " + c["text"][:400] for i, c in enumerate(ctx[:2])])
            return f"(Fallback) Model unavailable: {e}\n\nTop snippets:\n{snippet}\n\nAdd credits or switch provider."
        return f"(Fallback) Model unavailable: {e}\n\nAdd credits or switch provider."

def generate_general(message: str, history: List[Dict]):
    msgs = [{"role": "system", "content": "You are a helpful, friendly assistant."}]
    for h in history[-4:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": message})
    return chat_complete(msgs)

def confidence_from_hits(hits: List[Dict]):
    if not hits:
        return 0.2
    ds = [h.get("dist", 0.5) for h in hits]
    sim = float(np.exp(-float(np.mean(ds))))
    return max(0.3, min(0.95, sim))

def collection_stats():
    total = collection.count()
    nec = collection.get(where={"domain": "nec"}, include=["metadatas"], limit=1000000)
    watt = collection.get(where={"domain": "wattmonk"}, include=["metadatas"], limit=1000000)
    nec_titles = [m["title"] for m in nec.get("metadatas", [])][:5]
    watt_titles = [m["title"] for m in watt.get("metadatas", [])][:5]
    return {
        "total": int(total),
        "nec_count": len(nec.get("metadatas", [])),
        "wattmonk_count": len(watt.get("metadatas", [])),
        "nec_samples": nec_titles,
        "wattmonk_samples": watt_titles,
    }
