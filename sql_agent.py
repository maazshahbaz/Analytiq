# sql_agent_with_memory.py

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.llms import OpenAI
from langchain.agents import create_sql_agent, AgentType
from read_only_sql_tool import ReadOnlyQuerySQLDataBaseTool
from langchain.memory import ConversationBufferMemory

class SQLToolkitWrapper:
    def __init__(self, db):
        self.db = db
        self.dialect = db.dialect  
        self.read_only_tool = ReadOnlyQuerySQLDataBaseTool(db=db)

    def get_tools(self):
        return [self.read_only_tool]

def build_sql_agent(connection_string: str, temperature: float = 0.0):
    """
    Original builder (without memory). This function is optional if you want to
    support memory by default.
    """
    db = SQLDatabase.from_uri(connection_string)
    llm = OpenAI(temperature=temperature)
    toolkit = SQLToolkitWrapper(db=db)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return agent_executor

def build_sql_agent_with_memory(connection_string: str, temperature: float = 0.0):
    """
    Memory-enabled SQL agent builder using ConversationBufferMemory.
    """
    db = SQLDatabase.from_uri(connection_string)
    llm = OpenAI(temperature=temperature)
    toolkit = SQLToolkitWrapper(db=db)
    
    # Create a memory object to capture the conversation history.
    memory = ConversationBufferMemory(
         memory_key="chat_history",
         return_messages=True
    )
    
    agent_executor = create_sql_agent(
         llm=llm,
         toolkit=toolkit,
         verbose=True,
         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
         memory=memory  # Pass memory to the agent
    )
    return agent_executor



