import os
from pypdf import PdfReader
import docx
from rag import chunk, add_docs

def read_pdf(path: str) -> str:
    r = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in r.pages)

def read_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def load_and_ingest(domain: str, folder: str):
    docs = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if f.lower().endswith(".pdf"):
                text = read_pdf(p)
            elif f.lower().endswith(".docx"):
                text = read_docx(p)
            elif f.lower().endswith(".txt"):
                text = open(p, "r", encoding="utf-8", errors="ignore").read()
            else:
                continue
            if not text.strip():
                continue
            for idx, ck in enumerate(chunk(text)):
                docs.append({"id": f"{domain}:{f}:{idx}", "text": ck, "title": f, "source": p})
    if docs:
        add_docs(domain, docs)
