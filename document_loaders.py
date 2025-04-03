# document_loaders.py
import os
import re
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredWordDocumentLoader

# 1) A helper function to parse department/year from the filename
def parse_filename_for_metadata(filename: str) -> dict:
    """
    Example: tries to guess 'department' and 'year' from the filename.
    Replace this with your own logic or advanced parsing if needed.
    """
    filename_lower = filename.lower()

    # Guess department
    if "admissions" in filename_lower:
        department = "Admissions"
    elif "registrar" in filename_lower:
        department = "Registrar"
    elif "finance" in filename_lower:
        department = "Finance"
    else:
        department = "Unknown"

    # Guess year (looking for something like 2020, 2021, 2022, etc.)
    match = re.search(r"(20\d{2})", filename_lower)
    year = match.group(1) if match else "Unknown"

    return {"department": department, "year": year}

def load_pdf(file_obj, filename):
    """
    Extracts text from PDFs using PyPDF2, 
    and adds page_number + other metadata to each Document.
    """
    base_meta = parse_filename_for_metadata(filename)
    reader = PdfReader(file_obj)
    docs = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        meta = {
            "source": filename,
            "page_number": page_idx + 1,
        }
        meta.update(base_meta)  # merge in department, year, etc.
        doc = Document(page_content=text, metadata=meta)
        docs.append(doc)
    return docs

def load_docx(file_obj, filename):
    """
    Uses UnstructuredWordDocumentLoader to extract DOCX content,
    then merges parse_filename_for_metadata + source info.
    """
    base_meta = parse_filename_for_metadata(filename)
    temp_path = f"temp_{filename}"
    with open(temp_path, "wb") as f:
        f.write(file_obj.getvalue())

    loader = UnstructuredWordDocumentLoader(temp_path)
    docs = loader.load()  # returns a list of Documents
    for doc in docs:
        # Merge base metadata with existing doc.metadata
        combined_meta = doc.metadata.copy()
        combined_meta["source"] = filename
        combined_meta.update(base_meta)
        doc.metadata = combined_meta

    os.remove(temp_path)
    return docs

def load_excel(file_obj, filename):
    base_meta = parse_filename_for_metadata(filename)
    try:
        dfs = pd.read_excel(file_obj, sheet_name=None)
    except Exception:
        return []
    docs = []
    for sheet_name, df in dfs.items():
        if df.empty:
            continue
        headers = list(df.columns)
        
        sheet_text = f"Sheet: {sheet_name}\n"
        for idx, row in df.iterrows():
            row_desc = ", ".join([f"{col}: {row[col]}" for col in headers])
            sheet_text += row_desc + "\n"

        # Convert the list of headers into a string
        meta = {
            "source": filename,
            "sheet_name": sheet_name,
            "columns": ", ".join(headers),  # <--- fix here
            "row_count": len(df)
        }
        meta.update(base_meta)
        doc = Document(page_content=sheet_text, metadata=meta)
        docs.append(doc)
    return docs

def load_csv(file_obj, filename):
    base_meta = parse_filename_for_metadata(filename)
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        return []
    headers = list(df.columns)

    csv_text = "CSV Data:\n"
    for idx, row in df.iterrows():
        row_desc = ", ".join([f"{col}: {row[col]}" for col in headers])
        csv_text += row_desc + "\n"

    meta = {
        "source": filename,
        "columns": ", ".join(headers),  # <--- fix here
        "row_count": len(df)
    }
    meta.update(base_meta)
    doc = Document(page_content=csv_text, metadata=meta)
    return [doc]

# A single text splitter for all docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)
