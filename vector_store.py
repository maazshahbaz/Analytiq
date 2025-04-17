import os
from chromadb.config import Settings                # ← NEW
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import PERSIST_DIRECTORY, COLLECTION_NAME


def get_vector_store():
    """
    Return a Chroma vector‑store in classic single‑tenant mode.
    The explicit Settings() silences the “default_tenant” error introduced in
    chromadb v0.4.22+.
    """
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        client_settings=Settings(anonymized_telemetry=False),  # ← FIX
    )


# ------------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------------
def filtered_search(query: str, k: int, where: dict | None = None):
    """Similarity search with optional metadata filter."""
    vs = get_vector_store()
    return vs.similarity_search(query, k=k, filter=where or {})


def get_document_count():
    try:
        return get_vector_store()._collection.count()
    except Exception:
        return 0


def get_document_and_chunk_count():
    try:
        data = get_vector_store()._collection.get()
        total_chunks = len(data["documents"])
        total_docs = len(
            {m.get("source") for m in data["metadatas"] if "source" in m}
        )
        return total_docs, total_chunks
    except Exception:
        return 0, 0


def delete_file_vectors(filename):
    vs = get_vector_store()
    vs.delete(where={"source": filename})
    vs.persist()
