# scripts/run_pandas_agent.py
import sys, pathlib
# ensure project root is on PYTHONPATH
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from config.settings import OPENAI_API_KEY
from agents.pandas_agent.agent import build_pandas_agent_with_memory

import pandas as pd

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("❌ OPENAI_API_KEY must be set in your .env")

    # 1) Create a dummy DataFrame
    df = pd.DataFrame({
        "year":    [2020, 2021, 2022],
        "apples":  [10, 15, 12],
        "oranges": [7,  9,  14]
    })
    files = {"dummy.csv": {"Sheet1": df}}

    # 2) Build the agent
    agent = build_pandas_agent_with_memory(files, temperature=0.0)
    print("✅ Pandas Agent built successfully.")

    # 3) Test the describe tool
    desc = agent.run("Describe the data.")
    print("\n--- Describe Output ---\n")
    print(desc)

    # 4) Test a simple analytic query
    out = agent.run("What is the average of apples?")
    print("\n--- Average Query ---\n")
    print(out)

if __name__ == "__main__":
    main()
