# scripts/run_database_agent.py
import os
from pathlib import Path

# ensure repo root modules are on path
import sys; sys.path.append(str(Path(__file__).parent.parent))

from config.settings import PROJECT_ROOT, OPENAI_API_KEY
from agents.database_agent.agent import build_sql_agent_with_memory

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("❌ OPENAI_API_KEY is not configured!")

    # 1) Connect to demo SQLite DB
    db_file = PROJECT_ROOT / "data" / "my_database.db"
    conn_str = f"sqlite:///{db_file}"
    print("Using connection string:", conn_str)

    # 2) Build the agent
    agent = build_sql_agent_with_memory(conn_str, temperature=0.0)
    print("✅ SQL Agent built successfully.\n")

    # 3) List tables
    raw_tables = agent.run("SELECT name FROM sqlite_master WHERE type='table';")
    print("-- Raw tables response:")
    print(raw_tables, "\n")

    # 4) Parse—or just pick one known table:
    # Option A: hardcode
    sample_table = "admissions"

    # Option B: dynamic parse (uncomment if you like)
    # line = next((l for l in raw_tables.splitlines() if "," in l), "")
    # tables = [t.strip() for t in line.split(",")] if line else []
    # sample_table = tables[0] if tables else "admissions"

    print(f"-- Counting rows in table `{sample_table}`")
    count = agent.run(f"SELECT COUNT(*) AS row_count FROM {sample_table};")
    print(count)

if __name__ == "__main__":
    main()
