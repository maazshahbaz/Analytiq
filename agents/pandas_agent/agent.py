# pandas_agent/pandas_agent.py

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from .pandas_tools import make_describe_tool, make_plot_tool


def _flatten_sheets(files_dict):
    """Return list of dfs OR single df for create_pandas_dataframe_agent."""
    dfs = [df for sheets in files_dict.values() for df in sheets.values()]
    return dfs[0] if len(dfs) == 1 else dfs

def build_pandas_agent_with_memory(files_dict, temperature=0.0):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=_flatten_sheets(files_dict),
        agent_type="tool-calling",
        extra_tools=[
            make_describe_tool(files_dict),
            make_plot_tool(files_dict),
        ],
        number_of_head_rows=5,
        allow_dangerous_code=True,
        max_iterations=20,
        verbose=True,
    )
    return agent

def explain_dataframes(files_dict, agent):
    prompt = (
        "Generate a concise, reader‑friendly profile of each sheet, covering\n"
        "• column names & types • key stats • missing‑value counts • obvious anomalies.\n"
        "Finish with three insights or questions the user might explore next."
    )
    return agent.run(prompt)
