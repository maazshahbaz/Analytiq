# agents/unstructured_agent/document_loaders.py
import os, re
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
from agents.unstructured_agent.table_loaders import dataframe_to_docs
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def parse_filename_for_metadata(filename: str) -> dict:
    """Guess department & year from filename."""
    name = filename.lower()
    if "admissions" in name:
        dept = "Admissions"
    elif "registrar" in name:
        dept = "Registrar"
    elif "finance" in name:
        dept = "Finance"
    else:
        dept = "Unknown"
    yr_match = re.search(r"(20\d{2})", name)
    year = yr_match.group(1) if yr_match else "Unknown"
    return {"department": dept, "year": year}


def df_to_row_docs(df: pd.DataFrame, src: str, level: str, **base_meta):
    """Return one Document per row with explicit row / col lineage."""
    df.columns = [c.strip().lower() for c in df.columns]
    docs: list[Document] = []
    for ridx, row in df.iterrows():
        row_txt = "; ".join(f"{k}: {row[k]}" for k in df.columns)
        meta = {
            "source": src,
            "row_id": ridx,
            "columns": ",".join(df.columns),
            "level": level,  # 'row' or 'section'
            **base_meta,
        }
        docs.append(Document(page_content=row_txt, metadata=meta))
    return docs

# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
def load_pdf(file_obj, filename):
    base_meta = parse_filename_for_metadata(filename)
    reader = PdfReader(file_obj)
    docs = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        meta = {"source": filename, "page_number": i + 1, **base_meta}
        docs.append(Document(page_content=txt, metadata=meta))
    return docs


def load_docx(file_obj, filename):
    base_meta = parse_filename_for_metadata(filename)
    tmp = f"tmp_{filename}"
    with open(tmp, "wb") as f:
        f.write(file_obj.getvalue())
    loader = UnstructuredWordDocumentLoader(tmp)
    docs = loader.load()
    for d in docs:
        d.metadata = {"source": filename, **d.metadata, **base_meta}
    os.remove(tmp)
    return docs


def load_excel(file_obj, filename):
    base_meta = parse_filename_for_metadata(filename)
    # ▶  First try high‑fidelity loader that preserves cell structure
    try:
        loader = UnstructuredExcelLoader(file_obj, mode="elements")
        docs = loader.load()
        # Enrich metadata uniformly
        for d in docs:
            d.metadata.update({"source": filename, **base_meta})
        return docs
    except Exception:
        pass  # fall back to pandas if python‑docx/xlrd etc. fail

    # ▶  Fallback – pandas per‑row JSON docs
    try:
        dfs = pd.read_excel(file_obj, sheet_name=None)
    except Exception:
        return []

    docs: list[Document] = []
    for sheet, df in dfs.items():
        if df.empty:
            continue
        docs.extend(
            dataframe_to_docs(
                df,
                filename,
                sheet_name=sheet,
                level="row",
                **base_meta,
            )
        )
    return docs


def load_csv(file_obj, filename):
    base_meta = parse_filename_for_metadata(filename)
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        return []
    return dataframe_to_docs(df, filename, level="row", **base_meta)


# ------------------------------------------------------------------
# One splitter shared by all formats
# ------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, chunk_overlap=200, separators=["\n\n", "\n", " "]
)
