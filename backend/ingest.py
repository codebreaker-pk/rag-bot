import os
from pathlib import Path
from pypdf import PdfReader
import docx
from rag import chunk, add_docs

SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}

def read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
        return ""

def read_docx(path: str) -> str:
    try:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        print(f"[WARN] Failed to read DOCX {path}: {e}")
        return ""

def read_txt(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Failed to read TXT {path}: {e}")
        return ""

def load_and_ingest(domain: str, folder: str):
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"[ERROR] Folder not found: {folder}")
        return

    docs = []
    file_count = 0

    for root, _, files in os.walk(folder_path):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                continue

            file_path = os.path.join(root, f)
            file_count += 1

            if ext == ".pdf":
                text = read_pdf(file_path)
            elif ext == ".docx":
                text = read_docx(file_path)
            elif ext == ".txt":
                text = read_txt(file_path)
            else:
                continue

            if not text.strip():
                print(f"[INFO] Skipping empty file: {file_path}")
                continue

            for idx, ck in enumerate(chunk(text)):
                docs.append({
                    "id": f"{domain}:{f}:{idx}",
                    "text": ck,
                    "title": f,
                    "source": file_path
                })

    if not docs:
        print(f"[INFO] No valid documents found in {folder}")
        return

    # Batch upload to avoid huge payloads
    BATCH = 64
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i+BATCH]
        try:
            add_docs(domain, batch)
        except Exception as e:
            print(f"[ERROR] Failed to add_docs for batch {i // BATCH}: {e}")

    print(f"[DONE] Indexed {len(docs)} chunks from {file_count} files in '{folder}'")
