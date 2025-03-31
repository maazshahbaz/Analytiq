#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

import os
import streamlit as st
from io import BytesIO
import pandas as pd

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI

from PyPDF2 import PdfReader
from docx import Document

# For converting chat history into LangChain message objects
from langchain.schema import HumanMessage, AIMessage

# --- Config ---
os.environ["OPENAI_API_KEY"] = "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"  # 🔒 Use your .env file in production
persist_directory = os.path.join(os.getcwd(), "chroma_db")
collection_name = "institutional_docs"

# --- Document Extraction Helpers ---
def extract_text_from_pdf(file_obj):
    reader = PdfReader(file_obj)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def extract_text_from_docx(file_obj):
    doc = Document(file_obj)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_xlsx(file_obj):
    try:
        dfs = pd.read_excel(file_obj, sheet_name=None)
    except Exception as e:
        st.error(f"Excel read error: {str(e)}")
        return ""
    all_text = ""
    for sheet_name, df in dfs.items():
        if df.empty:
            continue
        headers = list(df.columns)
        sheet_text = f"Sheet: {sheet_name}\n"
        # Build a descriptive sentence for each row including headers
        for idx, row in df.iterrows():
            row_desc = ", ".join([f"{col}: {row[col]}" for col in headers])
            sheet_text += row_desc + "\n"
        all_text += sheet_text + "\n"
    return all_text

# --- Vector Store Helpers ---
def get_vector_store():
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
        collection_name=collection_name
    )

def delete_file_vectors(filename):
    vector_store = get_vector_store()
    vector_store.delete(where={"source": filename})
    vector_store.persist()


# --- Chat History Conversion Helper ---
def convert_history(history):
    """
    Convert structured chat history (list of dicts with keys "role" and "content")
    into a list of LangChain message objects.
    """
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

# --- UI Setup ---
st.set_page_config(page_title="Institutional Research Chat", layout="wide")
st.title("📊 Institutional Research AI Assistant")

# Initialize session state variables if not set
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # each item is a dict: {"role": "user"/"assistant", "content": "..."}

# --- Document Upload Section ---
st.markdown("### 📁 Upload Documents")
uploaded = st.file_uploader("Upload files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
if uploaded:
    with st.spinner("Processing uploaded files..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        vector_store = get_vector_store()
        for file in uploaded:
            # Skip if file already uploaded this session
            if file.name in [f["name"] for f in st.session_state.uploaded_files]:
                continue
            ext = file.name.split(".")[-1].lower()
            if ext == "pdf":
                text = extract_text_from_pdf(file)
            elif ext == "docx":
                text = extract_text_from_docx(file)
            elif ext == "xlsx":
                text = extract_text_from_xlsx(file)
            else:
                text = file.read().decode("utf-8", errors="ignore")
            chunks = text_splitter.split_text(text)
            metadatas = [{"source": file.name} for _ in chunks]
            if chunks:
                vector_store.add_texts(chunks, metadatas=metadatas)
                vector_store.persist()
                st.session_state.uploaded_files.append({"name": file.name})
        st.success("✅ Documents added to vector DB.")

# --- Document List & Deletion Section ---
st.markdown("### 🗂️ Uploaded Documents in Vector Store")
if st.session_state.uploaded_files:
    for file_entry in st.session_state.uploaded_files:
        col1, col2 = st.columns([5, 1])
        col1.write(file_entry["name"])
        if col2.button("🗑️ Remove", key=file_entry["name"]):
            delete_file_vectors(file_entry["name"])
            st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f["name"] != file_entry["name"]]
            st.success(f"Removed {file_entry['name']} from vector store.")
            st.experimental_rerun()
else:
    st.info("No documents in the vector store yet.")

# --- Conversational Chat Section ---
if st.session_state.uploaded_files:
    st.markdown("### 💬 Chat with Your Documents")

    vector_store = get_vector_store()
    llm = OpenAI(temperature=0)
    # Set up conversation memory with an explicit output_key to store only the final answer.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    # Update memory with existing chat history from session state.
    if st.session_state.chat_history:
        memory.chat_memory.messages = convert_history(st.session_state.chat_history)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    # Display existing chat history in a ChatGPT-style interface.
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
                        st.markdown(f"**Source {i+1}** — `{doc.metadata.get('source', 'Unknown')}`")
                        st.markdown(doc.page_content[:1000] + "...")
                        
# --- Optional: Clear Chat History ---
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
