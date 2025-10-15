# RAG Bot (Beginner-Friendly, Business-Ready)


> 📌 Live Demo Frontend: https://rag-bot-five.vercel.app/  
> 🛠️ Backend API: https://rag-bot-0qow.onrender.com

---

## Overview

Business RAG Bot is a **Retrieval-Augmented Generation (RAG)** chatbot focused on two domains: **NEC** (National Electrical Code) and **Wattmonk**.  
It can answer domain questions with **context citations**, small talk in “general” mode, and gracefully fallback if information is missing.

**Key features:**
- Ingests PDF / DOCX / TXT documents and indexes them in vector store  
- Domain classification (automatic or manual)  
- Citations: answers include `[1], [2]` style linking to source chunks  
- Confidence scoring  
- Two deployment parts:  
  - Backend with FastAPI + Chroma + Gemini embeddings  
  - Frontend with React, Vite, Material-UI  

---
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
   Invoke-RestMethod -Uri http://localhost:8000/ingest -Method POST
   Invoke-RestMethod -Uri http://localhost:8000/stats
   Invoke-RestMethod -Uri http://localhost:8000/chat -Method POST -ContentType 'application/json' -Body '{"message":"hello"}'
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
