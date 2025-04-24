# agents/database_agent/ui.py
import os
import streamlit as st

from config.settings import PROJECT_ROOT
from .create_test_db import create_and_populate_db
from .agent import build_sql_agent_with_memory
from utils.session_utils import (
    load_all_connection_strings,
    load_conversation,
    save_conversation,
    save_connection_string,
)

DB_PATH = PROJECT_ROOT / "data" / "my_database.db"

def run_ui(temperature):
    """🔍 SQL-Only Explorer + Chat"""
    st.subheader("SQL Explorer (Read-Only)")
    tab_conn, tab_chat = st.tabs(["Connections", "Chat"])

    # — Connections —
    with tab_conn:
        if not DB_PATH.exists():
            create_and_populate_db(DB_PATH)
            st.success(f"Created demo DB at {DB_PATH}")

        prev = load_all_connection_strings()
        sel = (
            st.selectbox("Pick saved connection:", prev)
            if prev else ""
        )
        conn_str = st.text_input(
            "SQLAlchemy URI:",
            value=sel or f"sqlite:///{DB_PATH}"
        )
        if st.button("🔗 Connect"):
            try:
                st.session_state.sql_agent = build_sql_agent_with_memory(
                    conn_str, temperature=temperature
                )
                st.session_state.current_connection = conn_str
                save_connection_string(conn_str)
                st.session_state.conversation = load_conversation(conn_str)
                st.success("Connected!")
            except Exception as e:
                st.error(f"❌ {e}")

    # — Chat —
    with tab_chat:
        if "current_connection" not in st.session_state:
            st.info("Connect first in the Connections tab.")
            return

        conv_box = st.empty()
        def render():
            md = ""
            for m in st.session_state.conversation:
                who = "User" if m["role"]=="user" else "SQL Agent"
                md += f"**{who}:** {m['content']}\n\n"
            return md

        conv_box.markdown(render())

        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            save_conversation(st.session_state.current_connection,
                              st.session_state.conversation)
            conv_box.markdown(render())

        q = st.text_area("Enter SQL or question:")
        if st.button("Send Query"):
            if q.strip():
                st.session_state.conversation.append(
                    {"role":"user","content":q}
                )
                conv_box.markdown(render())
                with st.spinner("Querying…"):
                    try:
                        res = st.session_state.sql_agent.run(q)
                        st.session_state.conversation.append(
                            {"role":"assistant","content":str(res)}
                        )
                        save_conversation(
                            st.session_state.current_connection,
                            st.session_state.conversation
                        )
                        conv_box.markdown(render())
                    except Exception as e:
                        st.error(f"❌ {e}")
            else:
                st.warning("Please enter a query.")
