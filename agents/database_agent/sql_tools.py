# sql_tools.py
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit as _BaseToolkit
from .read_only_sql_tool import ReadOnlyQuerySQLDataBaseTool
from .compare_tables_tool import make_compare_tables_tool

class SQLDatabaseToolkit(_BaseToolkit):
    def __init__(self, db: SQLDatabase, llm):
        super().__init__(db=db, llm=llm)

    @property
    def dialect(self) -> str:
        return str(self.db.dialect)

    def get_tools(self):
        # 1) keep ALL standard tools
        tools = super().get_tools()

        # 2) replace the write‑enabled query tool with read‑only version
        tools = [
            t for t in tools if t.name != "sql_db_query"
        ]  # strip default query tool
        tools.append(ReadOnlyQuerySQLDataBaseTool(db=self.db))

        # 3) add compare_tables helper (typed)
        tools.append(make_compare_tables_tool(self.db))

        return tools



def get_sql_toolkit(db: SQLDatabase, llm):
    return SQLDatabaseToolkit(db=db, llm=llm)
