# Backend (FastAPI)

## 1) Setup
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Open .env and paste your OpenAI key
```

## 2) Add your docs
Put your PDFs/DOCX/TXT here:
- `backend/data/nec/`
- `backend/data/wattmonk/`

## 3) Run the server
```bash
uvicorn main:app --reload --port 8000
```
Then, in another terminal, ingest your docs:
```bash
curl -X POST http://localhost:8000/ingest
```

Health check:
```
GET http://localhost:8000/health
```
