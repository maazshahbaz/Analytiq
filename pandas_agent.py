# pandas_agent.py

import pandas as pd
from langchain_openai import ChatOpenAI  # Adjust the import according to your package/version
from langchain.memory import ConversationBufferMemory

# ============================
# Helper Function: Generate a Summary for a DataFrame
# ============================
def generate_pandas_summary(df):
    """
    Generate a text summary for a Pandas DataFrame including shape, column types,
    descriptive statistics, and missing values.
    """
    summary = f"Shape: {df.shape}\n\n"
    summary += "Columns and Data Types:\n"
    summary += df.dtypes.to_string() + "\n\n"
    try:
        description = df.describe().to_string()
    except Exception:
        description = "No numeric columns to describe."
    summary += "Descriptive Statistics:\n" + description + "\n\n"
    summary += "Missing Values:\n" + df.isnull().sum().to_string() + "\n"
    return summary

# ============================
# Build the Pandas Agent with Buffer Memory (UPDATED)
# ============================
def build_pandas_agent_with_memory(files_dict, temperature=0.0):
    """
    Build a Pandas agent function with ConversationBufferMemory enabled.

    Arguments:
      files_dict (dict): Nested dictionary of:
        {
          "my_file.xlsx": {
             "Sheet1": <DataFrame>,
             "Sheet2": <DataFrame>
          },
          "my_file.csv": {
             "Sheet1": <DataFrame>
          }
        }
      temperature (float): Temperature parameter for the LLM.

    Returns:
      A function that takes a query string and returns the agent's response.
    """

    llm = ChatOpenAI(temperature=temperature)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # UPDATED CODE: Build context by iterating over each file and each sheet
    context = ""
    for file_name, sheets_dict in files_dict.items():
        for sheet_name, df in sheets_dict.items():
            context += f"File: '{file_name}', Sheet: '{sheet_name}'\n"
            context += generate_pandas_summary(df) + "\n"
            context += f"Preview (first 3 rows):\n{df.head(3).to_string(index=False)}\n\n"

    def pandas_agent(query: str) -> str:
        # Retrieve conversation history from memory.
        memory_vars = memory.load_memory_variables({})
        history = memory_vars.get("chat_history", [])
        history_str = ""
        for msg in history:
            if msg.get("role") == "human":
                history_str += f"User: {msg.get('content')}\n"
            else:
                history_str += f"Agent: {msg.get('content')}\n"

        # Construct the LLM prompt
        prompt = (
            f"Data context:\n{context}\n"
            f"Conversation history:\n{history_str}\n"
            f"Question: {query}\n"
            "Please provide a clear, concise answer based on the above context."
        )

        # Call the LLM and strip content
        response = llm(prompt).content.strip()
        # Save the conversation turn in memory
        memory.save_context({"input": query}, {"output": response})
        return response

    return pandas_agent

# ============================
# Generate an Initial Explanation of the DataFrames (UPDATED)
# ============================
def explain_dataframes(files_dict, agent):
    """
    Generate an initial explanation of the uploaded dataframes.

    Arguments:
      files_dict (dict): Same nested dict structure:
        {
          "my_file.xlsx": {
             "Sheet1": <DataFrame>,
             "Sheet2": <DataFrame>
          },
          "my_file.csv": {
             "Sheet1": <DataFrame>
          }
        }
      agent: The agent function returned by build_pandas_agent_with_memory.

    Returns:
      A string explanation from the agent.
    """
    context = ""
    # UPDATED CODE: Loop over each file and sheet
    for file_name, sheets_dict in files_dict.items():
        for sheet_name, df in sheets_dict.items():
            context += f"File: '{file_name}', Sheet: '{sheet_name}'\n"
            context += generate_pandas_summary(df) + "\n"
            context += f"Preview (first 3 rows):\n{df.head(3).to_string(index=False)}\n\n"

    # Build the explanation prompt
    explanation_prompt = (
        "Please provide a comprehensive explanation of the contents of the following file(s) and sheet(s). "
        "For each sheet, include:\n"
        "1. A list of columns with data types.\n"
        "2. Key descriptive statistics and summary of missing values.\n"
        "3. Any notable patterns or anomalies.\n"
        "4. Potential interpretations and uses of the data.\n\n"
        "File details:\n" + context
    )
    explanation = agent(explanation_prompt)
    return explanation
