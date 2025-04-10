#main.py
import os
import re
import json
import streamlit as st
import pandas as pd
from config import PERSIST_DIRECTORY
from document_loaders import (
    load_pdf, load_docx, load_excel, load_csv, text_splitter
)
from viewer import view_document
from langchain.docstore.document import Document
from vector_store import get_vector_store, delete_file_vectors, get_document_count
from vector_store import get_document_and_chunk_count
from hybrid_chain import HybridQAChain
import streamlit as st
from sql_agent import build_sql_agent_with_memory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.llms import OpenAI
from langchain.agents import create_sql_agent, AgentType
from read_only_sql_tool import ReadOnlyQuerySQLDataBaseTool
from langchain.memory import ConversationBufferMemory
from pandas_agent import build_pandas_agent_with_memory
from pandas_agent import build_pandas_agent_with_memory, explain_dataframes
from pandas_agent import (
    build_pandas_agent_with_memory,
    explain_dataframes
)
#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


openai_key = os.environ.get("OPENAI_API_KEY")

if not openai_key:
    st.error("❌ OPENAI_API_KEY is missing. Add it in Render → Environment.")
else:
    os.environ["OPENAI_API_KEY"] = openai_key

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
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "pandas_files" not in st.session_state:
    st.session_state.pandas_files = {} 
if "pandas_conversation" not in st.session_state:
    st.session_state.pandas_conversation = []

# Initialize the vector store once
vector_store = get_vector_store()


# File to store persistent conversation history associated with each connection string
CONVERSATION_FILE = "sql_conversations.json"

def load_all_connection_strings():
    """Load all previously used connection strings from the conversation file."""
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as f:
            all_conversations = json.load(f)
        return list(all_conversations.keys())
    return []

def load_conversation(connection_string):
    """Load conversation history for a given connection string."""
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as f:
            all_conversations = json.load(f)
        return all_conversations.get(connection_string, [])
    return []

def save_conversation(connection_string, conversation):
    """Persist the conversation history for the given connection string."""
    all_conversations = {}
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as f:
            all_conversations = json.load(f)
    all_conversations[connection_string] = conversation
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(all_conversations, f)

# ----------------
# Tabs for the App
# ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Document Management", 
    "💬 Chat", 
    "🔍 SQL & Data Explorer", 
    "📊 Pandas Agent"
])


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
        # Main chat container
        chat_container = st.container()
        
        # Chat input container - this will always be at the bottom
        input_container = st.container()
        
        # Bottom container for clear chat button
        bottom_container = st.container()
        
        # Display existing chat messages in the chat container
        with chat_container:
            # Keep track of all document buttons across different messages
            doc_button_count = 0
            
            for msg_idx, msg in enumerate(st.session_state.chat_history):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
                    # Only show sources for assistant messages
                    if msg["role"] == "assistant" and "sources" in msg:
                        sources = msg["sources"]
                        if sources:
                            with st.expander("📚 Source Documents"):
                                for i, doc in enumerate(sources):
                                    src = doc.metadata.get("source", "Unknown")
                                    st.markdown(f"**Source {i+1}** — `{src}`")
                                    st.markdown(f"- Department: {doc.metadata.get('department', 'N/A')}")
                                    st.markdown(f"- Year: {doc.metadata.get('year', 'N/A')}")
                                    # Show a snippet of the content
                                    snippet = doc.page_content[:500]
                                    st.markdown(snippet + ("..." if len(doc.page_content) > 500 else ""))
                                    
                                    # Create a unique key for each button using a global counter
                                    button_key = f"view_doc_{msg_idx}_{i}_{doc_button_count}"
                                    doc_button_count += 1
                                    
                                    if st.button("View Document", key=button_key):
                                        view_document(src, st.session_state.raw_files)
        
        # Input area always stays at the bottom of the conversation
        with input_container:
            user_input = st.chat_input("Ask something about the uploaded documents...")
            
            if user_input:
                # Immediately display user's message
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Add to session state
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                
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
                
                # Add assistant's response to history with sources
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
                
                # Force a rerun to update the UI with the new messages
                st.rerun()
        
        # Clear chat button at the very bottom
        with bottom_container:
            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()  # Force a rerun to update the UI

# ----------- 
# SQL Agent TAB (Multi-Tab Layout)
# -----------
with tab3:
    st.subheader("SQL Explorer (Read-Only)")

    # Create two sub-tabs inside the SQL & Data Explorer tab:
    sub_tab_conn, sub_tab_chat = st.tabs(["Connections", "Chat"])

    # ----------------------------
    # Sub-Tab 1: Connections
    # ----------------------------
    with sub_tab_conn:
        st.markdown("### Manage Connection")
        
        # List previously used connections (from persistent storage)
        previous_connections = load_all_connection_strings()
        if previous_connections:
            selected_connection = st.selectbox(
                "Previously used connections:",
                options=previous_connections,
                index=0
            )
        else:
            selected_connection = ""

        # Allow user to enter a new connection or use the selected one
        connection_string_input = st.text_input(
            "Enter SQLAlchemy connection string:",
            value=selected_connection if selected_connection else "",
            placeholder="sqlite:////full/path/to/database.db"
        )
        
        # Button to load or switch connection
        if st.button("Load Connection", key="load_conn_button"):
            if connection_string_input:
                try:
                    # Build the SQL agent using the memory-enabled builder.
                    st.session_state.sql_agent = build_sql_agent_with_memory(connection_string_input, temperature=0.0)
                    st.session_state.current_connection = connection_string_input
                    # Load existing conversation history if available.
                    st.session_state.conversation = load_conversation(connection_string_input)
                    st.success(f"Connection '{connection_string_input}' loaded successfully!")
                except Exception as e:
                    st.error(f"Could not connect: {e}")
            else:
                st.warning("Please enter a valid connection string.")


    # ----------------------------
    # Sub-Tab 2: Chat
    # ----------------------------
    # In the "Chat" sub-tab of the SQL Agent tab
with sub_tab_chat:
    if "current_connection" not in st.session_state:
        st.info("No connection loaded. Please go to the 'Connections' tab to load a connection.")
    else:
        # Create a placeholder for the conversation display.
        conversation_placeholder = st.empty()

        def render_conversation():
            """Render the conversation as a markdown string."""
            conversation_md = ""
            for message in st.session_state.conversation:
                role = message.get("role")
                content = message.get("content")
                if role == "user":
                    conversation_md += f"**User:** {content}\n\n"
                else:
                    conversation_md += f"**SQL Agent:** {content}\n\n"
            return conversation_md

        # Initially display the conversation.
        conversation_placeholder.markdown(render_conversation())

        # Option to clear the conversation.
        if st.button("Clear Conversation", key="clear_sql_conversation"):
            st.session_state.conversation = []
            save_conversation(st.session_state.current_connection, st.session_state.conversation)
            conversation_placeholder.markdown(render_conversation())
            st.success("Conversation cleared!")

        # Input field for new SQL query or question.
        user_query = st.text_area("Enter your SQL query or question:", key="sql_query_input")
        if st.button("Send Query", key="send_query"):
            if user_query:
                # Append the user's query to the conversation.
                st.session_state.conversation.append({"role": "user", "content": user_query})
                conversation_placeholder.markdown(render_conversation())  # Update display immediately.
                with st.spinner("Querying database..."):
                    try:
                        result = st.session_state.sql_agent.run(user_query)
                        # Append the agent's response.
                        st.session_state.conversation.append({"role": "assistant", "content": str(result)})
                        # Save updated conversation.
                        save_conversation(st.session_state.current_connection, st.session_state.conversation)
                        # Update conversation display with the new messages.
                        conversation_placeholder.markdown(render_conversation())
                    except Exception as e:
                        st.error(f"Error running query: {e}")
            else:
                st.warning("Please enter a query.")

# -----------
# PANDAS AGENT TAB (Multi-Tab Layout with Buffer Memory) - NEW CODE
# -----------
with tab4:
    st.subheader("Pandas Agent")
    sub_tab_files, sub_tab_chat = st.tabs(["Files", "Chat"])
    
    # --- Sub-Tab: Files ---
    with sub_tab_files:
        st.markdown("### Upload and Manage CSV/Excel Files")

        uploaded_pandas_files = st.file_uploader(
            "Upload CSV/Excel files",
            type=["csv", "xlsx"],
            accept_multiple_files=True
        )
        if uploaded_pandas_files:
            for file in uploaded_pandas_files:
                if file.name not in st.session_state.pandas_files:
                    try:
                        if file.name.lower().endswith('.csv'):
                            df = pd.read_csv(file)
                            # store as { "Sheet1": df } so everything is consistent
                            st.session_state.pandas_files[file.name] = {"Sheet1": df}
                        elif file.name.lower().endswith('.xlsx'):
                            df_sheets = pd.read_excel(file, sheet_name=None) 
                            st.session_state.pandas_files[file.name] = df_sheets
                    except Exception as e:
                        st.error(f"Error processing file {file.name}: {e}")
            st.success("Files uploaded successfully!")
            st.write("### Uploaded Files:")
            st.write(list(st.session_state.pandas_files.keys()))
        else:
            st.info("Please upload CSV or Excel files.")

        if st.session_state.pandas_files:
            selected_removals = st.multiselect(
                "Select files to remove",
                options=list(st.session_state.pandas_files.keys())
            )
            if st.button("Remove Selected Files"):
                for fname in selected_removals:
                    st.session_state.pandas_files.pop(fname, None)
                st.success("Selected files removed.")
    
    # --- Sub-Tab: Chat ---
    with sub_tab_chat:
        if not st.session_state.pandas_files:
            st.info("No files uploaded. Please upload files in the Files tab.")
        else:
            selected_file_names = st.multiselect(
                "Select one or two files for analysis",
                options=list(st.session_state.pandas_files.keys())
            )
            if not selected_file_names:
                st.info("Please select at least one file to proceed.")
            else:
                # selected_dataframes is {file_name: {sheet_name: DataFrame}}
                selected_dataframes = {fname: st.session_state.pandas_files[fname] for fname in selected_file_names}
                
                # Display each file's sheets
                if len(selected_dataframes) == 1:
                    file_key = selected_file_names[0]
                    st.markdown(f"### Previews for '{file_key}'")
                    sheets_dict = selected_dataframes[file_key]
                    for sheet_name, df in sheets_dict.items():
                        with st.expander(f"Preview of Sheet '{sheet_name}' in '{file_key}'"):
                            st.dataframe(df)
                else:
                    st.markdown("### Previews of Selected Files")
                    for fname, sheets_dict in selected_dataframes.items():
                        st.markdown(f"## File: '{fname}'")
                        for sheet_name, df in sheets_dict.items():
                            with st.expander(f"Preview of Sheet '{sheet_name}'"):
                                st.dataframe(df)
                
                # Build the memory-enabled agent
                pandas_agent = build_pandas_agent_with_memory(selected_dataframes, temperature=0.0)
                
                # Conversation placeholder
                conversation_placeholder = st.empty()
                def render_pandas_conversation():
                    conversation_md = ""
                    for msg in st.session_state.pandas_conversation:
                        role = msg.get("role")
                        content = msg.get("content")
                        if role == "user":
                            conversation_md += f"**User:** {content}\n\n"
                        else:
                            conversation_md += f"**Agent:** {content}\n\n"
                    return conversation_md
                
                # Automatic explanation if conversation is empty
                if not st.session_state.pandas_conversation:
                    with st.spinner("Generating detailed file explanation..."):
                        try:
                            explanation = explain_dataframes(selected_dataframes, pandas_agent)
                            st.session_state.pandas_conversation.append({"role": "agent", "content": explanation})
                        except Exception as e:
                            st.error(f"Error generating file explanation: {e}")
                    conversation_placeholder.markdown(render_pandas_conversation())
                
                # Follow-up queries
                user_query = st.text_area("Enter your query regarding the selected files:", key="pandas_query_input")
                if st.button("Send Query", key="send_query_pandas"):
                    if user_query:
                        st.session_state.pandas_conversation.append({"role": "user", "content": user_query})
                        conversation_placeholder.markdown(render_pandas_conversation())
                        with st.spinner("Analyzing..."):
                            try:
                                result = pandas_agent(user_query)
                                st.session_state.pandas_conversation.append({"role": "agent", "content": str(result)})
                                conversation_placeholder.markdown(render_pandas_conversation())
                            except Exception as e:
                                st.error(f"Error processing query: {e}")
                    else:
                        st.warning("Please enter a query.")

