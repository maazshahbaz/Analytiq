#config.py
import os

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)          # ← auto‑create

COLLECTION_NAME = "institutional_docs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
