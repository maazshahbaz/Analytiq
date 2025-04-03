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

def delete_file_vectors(filename):
    vector_store = get_vector_store()
    vector_store.delete(where={"source": filename})
    vector_store.persist()
