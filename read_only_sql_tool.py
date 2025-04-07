# read_only_sql_tool.py
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

class ReadOnlyQuerySQLDataBaseTool(QuerySQLDataBaseTool):
    """
    A read-only version of QuerySQLDataBaseTool.
    It checks if the query starts with SELECT or WITH,
    and raises an error otherwise.
    """

    def _run(self, query: str):
        safe_query = query.lower().strip()
        if not (safe_query.startswith("select") or safe_query.startswith("with")):
            raise ValueError("Only SELECT/WITH queries are allowed in read-only mode.")
        # Delegate to parent once we've verified
        return super()._run(query)
