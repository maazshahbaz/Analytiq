# agents/unstructured_agent/ui.py
import io
import streamlit as st
import pandas as pd
from langchain.docstore.document import Document

from config.settings import PERSIST_DIRECTORY
from .document_loaders import (
    load_pdf, load_docx, load_excel, load_csv, text_splitter
)
from .vector_store import (
    get_vector_store,
    delete_file_vectors,
    get_document_count,
    get_document_and_chunk_count,
)
from .agent import HybridQAChain
from .viewer import view_document

def run_ui(sheet_filter, temperature, top_k_vector, top_k_rerank):
    """📁 Upload & Manage Documents + 💬 Chat UI"""
    st.subheader("📁 Upload & Manage Documents")
    vector_store = get_vector_store()
    st.write(f"**Total documents in the vector store:** {get_document_count()}")

    uploaded = st.file_uploader(
        "Upload files (PDF, DOCX, XLSX, CSV)",
        type=["pdf", "docx", "xlsx", "csv"],
        accept_multiple_files=True
    )
    if uploaded:
        with st.spinner("Processing uploaded files..."):
            for file in uploaded:
                # avoid duplicates
                names = [f["name"] for f in st.session_state.uploaded_files]
                if file.name in names:
                    continue

                ext = file.name.rsplit(".", 1)[-1].lower()
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

                # cache any raw DataFrames
                if ext in ("xlsx", "csv"):
                    buf = io.BytesIO(file.getvalue())
                    if ext == "xlsx":
                        st.session_state.tables[file.name] = (
                            pd.read_excel(buf, sheet_name=None)
                        )
                    else:
                        st.session_state.tables[file.name] = pd.read_csv(buf)

                chunks = text_splitter.split_documents(docs)
                vector_store.add_documents(chunks, batch_size=128)
                vector_store.persist()

                st.session_state.uploaded_files.append({"name": file.name})

            st.success("✅ Documents added to vector DB.")
            st.write(f"**New total:** {get_document_count()}")

    # show list + buttons
    if st.session_state.uploaded_files:
        st.write("### Documents in Vector Store")
        for entry in st.session_state.uploaded_files:
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.write(f"**{entry['name']}**")
            if c2.button("🗑️", key=f"del_{entry['name']}"):
                delete_file_vectors(entry["name"])
                st.session_state.uploaded_files = [
                    e for e in st.session_state.uploaded_files
                    if e["name"] != entry["name"]
                ]
                st.success(f"Removed {entry['name']}")
            if c3.button("👁️", key=f"view_{entry['name']}"):
                view_document(entry["name"], st.session_state.raw_files)
    else:
        st.info("No documents uploaded yet.")

    # —————————————————————————————————————————————————————————
    st.markdown("---")
    st.subheader("💬 Chat with Your Documents")
    if not st.session_state.uploaded_files:
        st.warning("Upload at least one document first.")
        return

    # render chat history
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Sources"):
                    for doc in msg["sources"]:
                        src = doc.metadata.get("source", "Unknown")
                        st.markdown(f"- `{src}`")
                        snippet = doc.page_content[:300]
                        st.markdown(snippet + ("…" if len(doc.page_content) > 300 else ""))

    user_input = st.chat_input("Ask something about the docs…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking…"):
            chain = HybridQAChain(
                temperature=temperature,
                top_k_vector=top_k_vector,
                top_k_rerank=top_k_rerank,
                sheet_filter=sheet_filter or None,
            )
            result = chain.run(user_input)

        answer = result["answer"]
        sources = result.get("source_documents", [])
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
