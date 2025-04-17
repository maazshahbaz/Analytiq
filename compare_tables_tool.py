#compare_tables_tool.py
from typing import List
import pandas as pd
from sqlalchemy import text
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


# ---- Pydantic schema -------------------------------------------------
class CompareTablesArgs(BaseModel):
    table1: str = Field(..., description="First table name")
    table2: str = Field(..., description="Second table name")
    key: str = Field(..., description="Column that exists in both tables to join on")
    metrics: List[str] = Field(
        ..., description="List of SQL aggregate expressions, e.g. ['SUM(credits)']"
    )


# ---- Factory that returns a StructuredTool ---------------------------
def make_compare_tables_tool(db):
    """
    Build a StructuredTool so the agent can pass arguments by name.
    """

    def _compare(*, table1: str, table2: str, key: str, metrics: List[str]) -> str:
        # helper to build "SUM(col) AS SUM_col_A"
        def agg_list(alias: str) -> str:
            return ", ".join(
                f"{m} AS {m.replace('(', '_').replace(')', '')}_{alias}"
                for m in metrics
            )

        q1 = f"SELECT {key}, {agg_list('A')} FROM {table1} GROUP BY {key}"
        q2 = f"SELECT {key}, {agg_list('B')} FROM {table2} GROUP BY {key}"

        df1 = pd.read_sql(text(q1), db._engine)
        df2 = pd.read_sql(text(q2), db._engine)
        merged = df1.merge(df2, on=key, how="outer").fillna(0)

        return merged.to_markdown(index=False)

    return StructuredTool(
        name="compare_tables",
        description=(
            "Compare aggregates between two tables on a shared key.\n"
            "Arguments: table1, table2, key, metrics (list of aggregates)."
        ),
        func=_compare,
        args_schema=CompareTablesArgs,
        return_direct=False,
    )
