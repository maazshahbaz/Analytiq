# config.py
import os

# Set your API key and other constants here
os.environ["OPENAI_API_KEY"] = "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"
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "institutional_docs"
