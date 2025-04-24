# read_only_sql_tool.py   
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

class ReadOnlyQuerySQLDataBaseTool(QuerySQLDataBaseTool):
    """
    Drop‑in replacement for QuerySQLDataBaseTool that refuses
    any statement not starting with SELECT or WITH.
    """

    def _run(self, query: str):
        safe = query.lower().lstrip()
        if not (safe.startswith("select") or safe.startswith("with")):
            raise ValueError("Only SELECT / WITH statements are allowed in read‑only mode.")
        return super()._run(query)

    async def _arun(self, query: str):
        safe = query.lower().lstrip()
        if not (safe.startswith("select") or safe.startswith("with")):
            raise ValueError("Only SELECT / WITH statements are allowed in read‑only mode.")
        return await super()._arun(query)
