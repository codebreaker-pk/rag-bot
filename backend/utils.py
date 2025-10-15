import os
from uuid import uuid4
from dotenv import load_dotenv

# load .env when present (local dev)
load_dotenv()

def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def new_session() -> str:
    return str(uuid4())
