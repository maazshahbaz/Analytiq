# agents/unstructured_agent/table_loaders.py
import json, pandas as pd
from langchain.docstore.document import Document

def dataframe_to_docs(df: pd.DataFrame,
                      source: str,
                      sheet_name: str | None = None,
                      level: str = "row",
                      **base_meta):
    """Return one document per row + a sheet summary, using ONLY
    primitive metadata values so Chroma is happy."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # 1️⃣ stringify the dtypes dict
    dtypes_json = json.dumps({c: str(dt) for c, dt in df.dtypes.items()})   # 👈

    docs: list[Document] = []
    for idx, row in df.iterrows():
        docs.append(
            Document(
                page_content=json.dumps(row.to_dict(), ensure_ascii=False),
                metadata={
                    **base_meta,
                    "source": source,
                    "sheet_name": sheet_name,
                    "row_id": int(idx),
                    # 2️⃣ convert list -> comma‑sep string
                    "columns": ",".join(df.columns),                       
                    "dtypes": dtypes_json,                                 
                    "level": level,
                },
            )
        )

    summary = f"Sheet {sheet_name or 'data'} – {len(df)} rows. Columns: {', '.join(df.columns)}."
    docs.append(
        Document(
            page_content=summary,
            metadata={
                **base_meta,
                "source": source,
                "sheet_name": sheet_name,
                "level": "section",
            },
        )
    )
    return docs
