# sql_agent.py
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.llms import OpenAI
from langchain.agents import create_sql_agent, AgentType
from read_only_sql_tool import ReadOnlyQuerySQLDataBaseTool


class SQLToolkitWrapper:
    def __init__(self, db):
        self.db = db
        # Add the dialect attribute so that create_sql_agent can access it
        self.dialect = db.dialect  
        self.read_only_tool = ReadOnlyQuerySQLDataBaseTool(db=db)

    def get_tools(self):
        return [self.read_only_tool]


def build_sql_agent(connection_string: str, temperature: float = 0.0):
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

