# RAG Bot (Beginner-Friendly, Business-Ready)

## Quick Start (Windows or Linux)
1. **Backend**
   ```bash
   cd backend
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Linux/Mac: . .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env   # paste your OpenAI key
   uvicorn main:app --reload --port 8000
   ```
   In another terminal:
   ```bash
   curl -X POST http://localhost:8000/ingest
   ```

2. **Frontend**
   ```bash
   cd frontend
   npm install
   cp .env.example .env
   npm run dev
   ```
   Open: http://localhost:5173

## Where to drop documents
- `backend/data/nec/` and `backend/data/wattmonk/` (PDF/DOCX/TXT). Then run `/ingest` again.

## Notes
- Local vector DB: Chroma (folder `backend/vectorstore`)
- Models: OpenAI (gpt-4o-mini + text-embedding-3-small)
- You can later swap Chroma → Pinecone, memory → Redis, add auth, and deploy.
