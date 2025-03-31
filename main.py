#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


import os
import streamlit as st
from io import BytesIO
import pandas as pd

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI

from PyPDF2 import PdfReader
from docx import Document

# --- Config ---
os.environ["OPENAI_API_KEY"] = "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"  # 🔒 Use env file in production
persist_directory = os.path.join(os.getcwd(), "chroma_db")
collection_name = "institutional_docs"

# --- Helpers ---
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
    return "\n".join([f"{name}\n{df.to_csv(index=False)}" for name, df in dfs.items()])

def get_vector_store():
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
        collection_name=collection_name
    )

def delete_file_vectors(filename):
    vector_store = get_vector_store()
    vector_store.delete(filter={"source": filename})
    vector_store.persist()

# --- UI Setup ---
st.set_page_config(page_title="Institutional Research Chat", layout="wide")
st.title("📊 Institutional Research AI Assistant")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Upload ---
st.markdown("### 📁 Upload Documents")
uploaded = st.file_uploader("Upload files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
if uploaded:
    with st.spinner("Processing uploaded files..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        vector_store = get_vector_store()

        for file in uploaded:
            if file.name in [f["name"] for f in st.session_state.uploaded_files]:
                continue  # Already uploaded this session

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

# --- File List + Delete ---
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

# --- Conversational Chat ---
# --- Conversational Chat ---
if st.session_state.uploaded_files:
    st.markdown("### 💬 Chat with Your Documents")

    vector_store = get_vector_store()
    llm = OpenAI(temperature=2)
    
    # Set output_key to "answer" so that memory knows which output to store
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    for i, msg in enumerate(st.session_state.chat_history):
        role = "user" if i % 2 == 0 else "assistant"
        with st.chat_message(role):
            st.markdown(msg)

    user_input = st.chat_input("Ask something about the uploaded documents...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            result = qa_chain({"question": user_input})

        answer = result["answer"]
        sources = result.get("source_documents", [])

        st.session_state.chat_history.append(user_input)
        st.session_state.chat_history.append(answer)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}** — `{doc.metadata.get('source', 'Unknown')}`")
                        st.markdown(doc.page_content[:1000] + "...")

# --- Optional: Reset Chat ---
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
