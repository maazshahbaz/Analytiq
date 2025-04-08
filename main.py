#main.py
import os
import streamlit as st
from config import PERSIST_DIRECTORY
from document_loaders import (
    load_pdf, load_docx, load_excel, load_csv, text_splitter
)
from viewer import view_document
from langchain.docstore.document import Document
from vector_store import get_vector_store, delete_file_vectors, get_document_count
from vector_store import get_document_and_chunk_count
# Import Hybrid chain
from hybrid_chain import HybridQAChain
import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.llms import OpenAI
from sql_agent import build_sql_agent
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Institutional Research Chat",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("📊 Institutional Research AI Assistant")
st.write(
    "Welcome! Upload documents, chat with them, and easily find the info you need."
)

# --------------------
# Sidebar - Parameters
# --------------------
st.sidebar.header("Advanced Chat Settings")
temperature = st.sidebar.slider(
    "LLM Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Higher temperature = more creative, random answers",
)
top_k_vector = st.sidebar.number_input(
    "Vector DB K (top_k_vector)",
    min_value=1,
    max_value=50,
    value=10,
    help="Number of embeddings to retrieve from the vector store",
)
top_k_rerank = st.sidebar.number_input(
    "Re-rank K (top_k_rerank)",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of documents to re-rank after vector retrieval",
)

# ------------
# Session State
# ------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "raw_files" not in st.session_state:
    st.session_state.raw_files = {}

# Initialize the vector store once
vector_store = get_vector_store()

# ----------------
# Tabs for the App
# ----------------
tab1, tab2, tab3 = st.tabs(["📁 Document Management", "💬 Chat", "🔍 SQL & Data Explorer"])


# ---------------
# DOCUMENT TAB
# ---------------
with tab1:
    st.subheader("Upload and Manage Documents")

    # 1) Show the existing document count at the top of this section
    doc_count = get_document_count()  # <-- Make sure you import this from vector_store
    st.write(f"**Total documents in the vector store:** {doc_count}")

    uploaded = st.file_uploader(
        "Upload files (PDF, DOCX, XLSX, CSV)",
        type=["pdf", "docx", "xlsx", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded:
        with st.spinner("Processing uploaded files..."):
            for file in uploaded:
                st.session_state.raw_files[file.name] = file.getvalue()

                # Avoid re-upload duplication
                if file.name in [f["name"] for f in st.session_state.uploaded_files]:
                    continue

                ext = file.name.split(".")[-1].lower()
                docs = []
                # Add try-except if you want better error handling:
                # try:
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

                # Split each doc into smaller chunks (if needed)
                doc_chunks = text_splitter.split_documents(docs)

                # Add them to the vector store with all enriched metadata
                vector_store.add_documents(doc_chunks)
                vector_store.persist()

                # Keep track of uploaded file
                st.session_state.uploaded_files.append({"name": file.name})

            st.success("✅ Documents added to vector DB.")

            # 2) Re-check the doc count after upload
            doc_count = get_document_count()
            st.write(f"**Updated documents in the vector store:** {doc_count}")

    st.write("### Documents in Vector Store")
    if st.session_state.uploaded_files:
        # Create a table-like layout
        for file_entry in st.session_state.uploaded_files:
            col1, col2, col3 = st.columns([4, 1, 1])
            col1.write(f"**{file_entry['name']}**")
            remove_button = col2.button("🗑️", key=f"remove_{file_entry['name']}")
            view_button = col3.button("👁️", key=f"view_{file_entry['name']}")

            if remove_button:
                delete_file_vectors(file_entry["name"])
                st.session_state.uploaded_files = [
                    f
                    for f in st.session_state.uploaded_files
                    if f["name"] != file_entry["name"]
                ]
                if file_entry["name"] in st.session_state.raw_files:
                    del st.session_state.raw_files[file_entry["name"]]
                st.success(f"Removed {file_entry['name']} from vector store.")

                # Optionally show doc count again after removal
                doc_count, chunk_count = get_document_and_chunk_count()
                st.write(f"**Total documents in the vector store:** {doc_count}")
                st.write(f"**Total chunks in the vector store:** {chunk_count}")

            if view_button:
                view_document(file_entry["name"], st.session_state.raw_files)
    else:
        st.info("No documents have been uploaded yet.")

# -----------
# CHAT TAB
# -----------
with tab2:
    st.subheader("Chat with Your Documents")
    if not st.session_state.uploaded_files:
        st.warning("Please upload at least one document to begin chatting.")
    else:
        # Display existing chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # The input widget for new questions
        user_input = st.chat_input("Ask something about the uploaded documents...")
        if user_input:
            # Immediately display user's message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Create a new chain with user-selected settings
            hybrid_chain = HybridQAChain(
                temperature=temperature,
                top_k_vector=top_k_vector,
                top_k_rerank=top_k_rerank
            )

            with st.spinner("Thinking..."):
                result = hybrid_chain.run(user_input)

            answer = result["answer"]
            sources = result.get("source_documents", [])

            # Update session state
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

            # Display assistant's answer
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(sources):
                            src = doc.metadata.get("source", "Unknown")
                            st.markdown(f"**Source {i+1}** — `{src}`")
                            st.markdown(
                                f"- Department: {doc.metadata.get('department', 'N/A')}"
                            )
                            st.markdown(
                                f"- Year: {doc.metadata.get('year', 'N/A')}"
                            )
                            # Show a snippet of the content
                            snippet = doc.page_content[:500]
                            st.markdown(snippet + ("..." if len(doc.page_content) > 500 else ""))
                            if st.button("View Document", key=f"view_src_{i}_{src}"):
                                view_document(src, st.session_state.raw_files)

        # Button to clear chat history
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")


# ----------- 
# SQL Agent TAB
# -----------
with tab3:
    st.subheader("SQL Explorer (Read-Only)")

    connection_string = st.text_input(
        "Enter SQLAlchemy connection string:",
        placeholder="sqlite:////home/ashfaq93/SST/Analytics_Tool/my_database.db"
    )
    
    if connection_string:
        if "sql_agent" not in st.session_state:
            try:
                st.session_state.sql_agent = build_sql_agent(connection_string, temperature=0.0)
                st.success("SQL Agent connected successfully!")
            except Exception as e:
                st.error(f"Could not connect to database: {e}")
        else:
            st.info("SQL Agent is already created. You can run queries below.")

        user_query = st.text_area("Enter a question or query for your SQL database:")

        if st.button("Run Query"):
            if user_query:
                with st.spinner("Querying database..."):
                    try:
                        result = st.session_state.sql_agent.run(user_query)
                        st.markdown("### Result")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Error running query: {e}")
    else:
        st.warning("Please enter a valid connection string.")