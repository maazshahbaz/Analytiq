# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# 1) load a .env file at project root (you'll create this next)
load_dotenv()

# 2) where to persist your vector DB
PROJECT_ROOT = Path(__file__).parent.parent
PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"
PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

# 3) collection name
COLLECTION_NAME = "institutional_docs"

# 4) your OpenAI key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
