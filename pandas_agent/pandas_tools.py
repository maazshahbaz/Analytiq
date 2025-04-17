# pandas_tools.py

from typing import List, Optional
from pydantic import BaseModel, Field, root_validator
import pandas as pd
from langchain.tools import StructuredTool
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────── describe_df ───────────────────
class DescribeArgs(BaseModel):
    file: Optional[str] = Field(
        None, description="Filename as uploaded (omit if only one file)"
    )
    sheet: Optional[str] = Field(
        None, description="Sheet name (default: first sheet)"
    )

    @root_validator(pre=True)
    def allow_missing_file_if_single(cls, values):
        # nothing to do here; we’ll handle it in the function
        return values

def make_describe_tool(files_dict: dict):
    filenames = list(files_dict.keys())

    def _describe(file: Optional[str] = None, sheet: Optional[str] = None) -> str:
        # 1) pick the only file if none specified
        if file is None:
            if len(filenames) == 1:
                file = filenames[0]
            else:
                raise ValueError("Must specify a file when multiple are loaded.")

        # 2) pick the first sheet if none specified
        sheets = list(files_dict[file].keys())
        target_sheet = sheet or sheets[0]

        df = files_dict[file][target_sheet]
        return df.describe(include="all", datetime_is_numeric=True).to_markdown()

    return StructuredTool(
        name="describe_df",
        description=(
            "Return descriptive statistics for a sheet. "
            "Args: file (optional if only one file), sheet (optional)."
        ),
        func=_describe,
        args_schema=DescribeArgs,
    )

# ─────────────────── plot_df (bar) ───────────────────
class PlotArgs(BaseModel):
    file: Optional[str] = Field(
        None, description="Filename as uploaded (omit if only one file)"
    )
    sheet: Optional[str] = Field(
        None, description="Sheet name (default: first sheet)"
    )
    x: str = Field(..., description="Column for x-axis")
    y: str = Field(..., description="Column for y-axis")

def make_plot_tool(files_dict: dict):
    filenames = list(files_dict.keys())

    def _plot(
        file: Optional[str] = None,
        sheet: Optional[str] = None,
        x: str = None,
        y: str = None
    ) -> str:
        # file
        if file is None:
            if len(filenames) == 1:
                file = filenames[0]
            else:
                raise ValueError("Must specify a file when multiple are loaded.")

        # sheet
        sheets = list(files_dict[file].keys())
        target_sheet = sheet or sheets[0]

        df = files_dict[file][target_sheet]

        # do the plot
        fig, ax = plt.subplots()
        df.plot(kind="bar", x=x, y=y, ax=ax)
        ax.set_title(f"{y} vs {x}")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        return f"Displayed bar chart of `{y}` vs `{x}` for file `{file}` sheet `{target_sheet}`."

    return StructuredTool(
        name="plot_df",
        description=(
            "Create a bar chart of y vs x and display it in Streamlit. "
            "Args: file (optional if only one file), sheet (optional), x, y."
        ),
        func=_plot,
        args_schema=PlotArgs,
    )
