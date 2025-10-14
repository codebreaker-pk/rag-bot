import os, uuid
from dotenv import load_dotenv
load_dotenv()

def get_env(k, default=None):
    return os.getenv(k, default)

def new_session():
    return str(uuid.uuid4())
