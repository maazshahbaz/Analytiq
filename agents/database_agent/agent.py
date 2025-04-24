# agents/database_agent/agent.py
import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent, AgentType
from langchain.memory import ConversationBufferMemory

from config.settings import OPENAI_API_KEY
from .sql_tools import get_sql_toolkit

# ---------------------------------------------------------------------
# Dialect‑specific cheat‑sheet
# ---------------------------------------------------------------------
DIALECT_HINTS = {
    "sqlite": """
SQLite has no built‑in STDDEV/STDDEVP function.
To compute standard deviation σ, either call the custom `stdev()` UDF
already registered, or use a CTE:

WITH stats AS (
  SELECT AVG({col}) AS mean,
         SQRT(AVG( ({col} - AVG({col})) * ({col} - AVG({col})) )) AS stdev
  FROM {table}
)
SELECT * FROM {table}, stats
WHERE {table}.{col} > stats.mean + stats.stdev;
"""
    # add more edge cases if needed
}

BASE_PROMPT = """
You are a data‑analytics assistant.

• Think step‑by‑step before generating SQL.
• For two‑table comparisons, inspect schemas first and use `compare_tables`.
• Always LIMIT result sets to 1 000 rows unless the user insists otherwise.
• Prefer aggregates (COUNT, AVG, SUM, MIN, MAX) over raw dumps.
• If a query fails, fix and retry.
• All operations are read‑only (no INSERT/UPDATE/DELETE).
"""


def build_sql_agent_with_memory(connection_string: str, temperature: float = 0.0):
    try:
        # 1) Connect to the DB -------------------------------------------------
        db = SQLDatabase.from_uri(connection_string)
        dialect = str(db.dialect).lower()

        # 2) Optional: register stdev() for SQLite ----------------------------
        if dialect == "sqlite":
            def _stdev(ctx, value):
                # ctx.rows : list[(float,)]
                vals = [float(v[0]) for v in ctx]
                return statistics.pstdev(vals)
            raw_conn = db._engine.raw_connection()
            raw_conn.create_aggregate("stdev", 1, _stdev)
            raw_conn.close()

        # 3) Create LLM -------------------------------------------------------
        llm = ChatOpenAI(
            temperature=temperature,
            model_name="gpt-3.5-turbo",
            api_key=st.secrets["OPENAI_API_KEY"],
        )

        # 4) Build toolkit and memory ----------------------------------------
        toolkit = get_sql_toolkit(db, llm)
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # 5) Compose dynamic system prompt -----------------------------------
        dialect_note = DIALECT_HINTS.get(dialect, "")
        system_prompt = BASE_PROMPT + dialect_note

        # 6) Create agent (function‑calling mode) -----------------------------
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            memory=memory,
            verbose=True,
            system_message=system_prompt,
            handle_parsing_errors=True,   # auto‑retry malformed outputs
            max_iterations=25,            # give the agent more breathing room
        )

        # 7) Warm‑cache schema into memory -----------------------------------
        schema_text = db.get_table_info()
        memory.save_context(
            {"input": "Show me the database schema"},
            {"output": f"Database schema:\n{schema_text}"},
        )

        return agent

    except Exception as e:
        import traceback
        err, tb = str(e), traceback.format_exc()
        raise Exception(f"Error building SQL agent: {err}\n\nDetails:\n{tb}")
