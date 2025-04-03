# main.py
import os
import streamlit as st
from config import PERSIST_DIRECTORY  # For example
from document_loaders import (
    load_pdf, load_docx, load_excel, load_csv, text_splitter
)
from vector_store import get_vector_store, delete_file_vectors
from chat import create_qa_chain, convert_history
from viewer import view_document
from langchain.docstore.document import Document

# --- UI Setup ---
st.set_page_config(page_title="Institutional Research Chat", layout="wide")
st.title("📊 Institutional Research AI Assistant")

# Initialize session state variables if not already set
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # List of dicts with key "name"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Each item: {"role": "user"/"assistant", "content": "..."}
if "raw_files" not in st.session_state:
    st.session_state.raw_files = {}  # Dict: {filename: file bytes}

# --- Document Upload Section ---
st.markdown("### 📁 Upload Documents")
uploaded = st.file_uploader("Upload files", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
if uploaded:
    with st.spinner("Processing uploaded files..."):
        vector_store = get_vector_store()
        for file in uploaded:
            st.session_state.raw_files[file.name] = file.getvalue()
            if file.name in [f["name"] for f in st.session_state.uploaded_files]:
                continue
            ext = file.name.split(".")[-1].lower()
            docs = []
            if ext == "pdf":
                docs = load_pdf(file, file.name)
            elif ext == "docx":
                docs = load_docx(file, file.name)
            elif ext == "xlsx":
                docs = load_excel(file, file.name)
            elif ext == "csv":
                docs = load_csv(file, file.name)
            else:
                text = file.read().decode("utf-8", errors="ignore")
                docs = [Document(page_content=text, metadata={"source": file.name})]
            
            if docs:
                doc_chunks = text_splitter.split_documents(docs)
                vector_store.add_texts(
                    [doc.page_content for doc in doc_chunks],
                    metadatas=[doc.metadata for doc in doc_chunks]
                )
                vector_store.persist()
                st.session_state.uploaded_files.append({"name": file.name})
        st.success("✅ Documents added to vector DB.")

# --- Document List & Deletion Section ---
st.markdown("### 🗂️ Uploaded Documents in Vector Store")
if st.session_state.uploaded_files:
    for file_entry in st.session_state.uploaded_files:
        col1, col2, col3 = st.columns([4, 1, 2])
        col1.write(file_entry["name"])
        if col2.button("🗑️ Remove", key=file_entry["name"]):
            delete_file_vectors(file_entry["name"])
            st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f["name"] != file_entry["name"]]
            st.success(f"Removed {file_entry['name']} from vector store.")
            if file_entry["name"] in st.session_state.raw_files:
                del st.session_state.raw_files[file_entry["name"]]
        if col3.button("👁️ View", key=f"view_{file_entry['name']}"):
            view_document(file_entry["name"], st.session_state.raw_files)
else:
    st.info("No documents in the vector store yet.")

# --- Conversational Chat Section ---
if st.session_state.uploaded_files:
    st.markdown("### 💬 Chat with Your Documents")
    vector_store = get_vector_store()
    qa_chain = create_qa_chain(st.session_state.chat_history)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input = st.chat_input("Ask something about the uploaded documents...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Thinking..."):
            result = qa_chain({"question": user_input})
        answer = result["answer"]
        sources = result.get("source_documents", [])
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(sources):
                        src = doc.metadata.get("source", "Unknown")
                        st.markdown(f"**Source {i+1}** — `{src}`")
                        st.markdown(doc.page_content[:1000] + "...")
                        if st.button("View Document", key=f"view_src_{i}_{src}"):
                            view_document(src, st.session_state.raw_files)

# --- Clear Chat History ---
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
