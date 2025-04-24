# main.py
import os
import streamlit as st

from config.settings import STREAMLIT_PAGE_TITLE, OPENAI_API_KEY
from agents.unstructured_agent.ui import run_ui as run_docs_ui
from agents.database_agent.ui import run_ui as run_sql_ui
from agents.pandas_agent.ui import run_ui as run_pandas_ui

def main():
    # ──────── 1. Ensure API key ────────
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY is not set in config/settings.py")
        st.stop()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # ──────── 2. Session-state defaults ────────
    st.session_state.setdefault("uploaded_files", [])
    st.session_state.setdefault("raw_files", {})
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("conversation", [])
    st.session_state.setdefault("pandas_files", {})
    st.session_state.setdefault("pandas_conversation", [])
    st.session_state.setdefault("tables", {})

    # ──────── 3. Page config & sidebar ────────
    st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, layout="wide")
    st.title(STREAMLIT_PAGE_TITLE)

    sheet_filter = st.sidebar.text_input(
        "Restrict search to sheet (optional)",
        placeholder="e.g., Admissions_23"
    )
    temperature = st.sidebar.slider(
        "LLM Temperature", 0.0, 1.0, 0.0, step=0.1
    )
    top_k_vector = st.sidebar.number_input(
        "Vector DB K", min_value=1, max_value=50, value=10
    )
    top_k_rerank = st.sidebar.number_input(
        "Re-rank K", min_value=1, max_value=10, value=3
    )

    # ──────── 4. Top-level tabs ────────
    tab_docs, tab_sql, tab_pandas = st.tabs([
        "📁 Docs & Chat",
        "🔍 SQL Explorer",
        "🐼 Pandas Agent",
    ])

    with tab_docs:
        run_docs_ui(sheet_filter, temperature, top_k_vector, top_k_rerank)

    with tab_sql:
        run_sql_ui(temperature)

    with tab_pandas:
        run_pandas_ui(temperature)

if __name__ == "__main__":
    main()
