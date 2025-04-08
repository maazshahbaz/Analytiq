# config.py
import os

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "institutional_docs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
