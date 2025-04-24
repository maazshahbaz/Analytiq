# agents/pandas_agent/ui.py
import streamlit as st
import pandas as pd

from .agent import build_pandas_agent_with_memory, explain_dataframes

def run_ui(temperature):
    """📊 Upload & Chat with DataFrames"""
    st.subheader("Pandas Agent")

    tab_files, tab_chat = st.tabs(["Files", "Chat"])

    # — Files tab —
    with tab_files:
        ups = st.file_uploader(
            "Upload CSV / Excel",
            type=["csv", "xlsx"],
            accept_multiple_files=True
        )
        if ups:
            for f in ups:
                name = f.name
                if name in st.session_state.pandas_files:
                    continue
                try:
                    if name.lower().endswith(".csv"):
                        df = pd.read_csv(f, na_values=["NA","n/a","---"])
                        st.session_state.pandas_files[name] = {"Sheet1": df}
                    else:
                        sheets = pd.read_excel(
                            f, sheet_name=None, na_values=["NA","n/a","---"]
                        )
                        for k,df in sheets.items():
                            sheets[k] = df
                        st.session_state.pandas_files[name] = sheets
                except Exception as e:
                    st.error(f"Error loading {name}: {e}")
            st.success("Files loaded!")
        if st.session_state.pandas_files:
            rem = st.multiselect(
                "Remove files", list(st.session_state.pandas_files)
            )
            if st.button("Remove"):
                for r in rem:
                    st.session_state.pandas_files.pop(r, None)
                st.success("Removed.")

    # — Chat tab —
    with tab_chat:
        if not st.session_state.pandas_files:
            st.info("Upload files first in the Files tab.")
            return

        sel = st.multiselect(
            "Select 1–2 files to analyze",
            list(st.session_state.pandas_files)
        )
        if not sel:
            st.info("Please select at least one file.")
            return

        dfs = {n: st.session_state.pandas_files[n] for n in sel}
        # previews
        for fname, sheets in dfs.items():
            st.markdown(f"#### `{fname}` Previews")
            for sname, df in sheets.items():
                with st.expander(f"{sname}"):
                    st.dataframe(df)

        agent = build_pandas_agent_with_memory(dfs, temperature=temperature)
        chat_box = st.empty()

        def render_convo():
            txt = ""
            for m in st.session_state.pandas_conversation:
                tag = "User" if m["role"]=="user" else "Agent"
                txt += f"**{tag}:** {m['content']}\n\n"
            return txt

        if not st.session_state.pandas_conversation:
            with st.spinner("Explaining data…"):
                expl = explain_dataframes(dfs, agent)
                st.session_state.pandas_conversation.append(
                    {"role":"agent","content":expl}
                )
        chat_box.markdown(render_convo())

        qry = st.text_area("Ask about your data:")
        if st.button("Send"):
            if qry.strip():
                st.session_state.pandas_conversation.append(
                    {"role":"user","content":qry}
                )
                chat_box.markdown(render_convo())
                with st.spinner("Analyzing…"):
                    res = agent(qry)
                    st.session_state.pandas_conversation.append(
                        {"role":"agent","content":str(res)}
                    )
                    chat_box.markdown(render_convo())
            else:
                st.warning("Enter a question.")
