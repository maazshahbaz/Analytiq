# vector_store.py
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import PERSIST_DIRECTORY, COLLECTION_NAME

def get_vector_store():
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )

def get_document_count():
    """Return the number of documents in the Chroma collection."""
    try:
        vector_store = get_vector_store()
        return vector_store._collection.count()
    except Exception as e:
        print(f"Error getting document count: {e}")
        return 0

def get_document_and_chunk_count():
    """
    Returns two integers:
      - total_docs: The number of unique source documents in the Chroma collection
      - total_chunks: The total number of chunks in the collection
    """
    try:
        vector_store = get_vector_store()
        # By default, this returns a dict with keys: ids, embeddings, metadatas, documents, etc.
        all_data = vector_store._collection.get()
        
        # total_chunks = number of entries (rows) in the collection
        total_chunks = len(all_data["documents"])
        
        # Identify unique source filenames from the metadata
        unique_sources = set()
        for meta in all_data["metadatas"]:
            if "source" in meta:
                unique_sources.add(meta["source"])
        
        total_docs = len(unique_sources)
        return total_docs, total_chunks
    except Exception as e:
        print(f"Error getting doc/chunk count: {e}")
        return 0, 0

def delete_file_vectors(filename):
    vector_store = get_vector_store()
    vector_store.delete(where={"source": filename})
    vector_store.persist()
