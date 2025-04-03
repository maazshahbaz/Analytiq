# document_loaders.py
import os
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader  # Simple PDF extraction without OCR

from langchain.docstore.document import Document

def load_pdf(file_obj, filename):
    """
    Extracts text from PDFs using PyPDF2.
    """
    reader = PdfReader(file_obj)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    doc = Document(page_content=text, metadata={"source": filename})
    return [doc]

def load_docx(file_obj, filename):
    """
    Uses UnstructuredWordDocumentLoader to extract DOCX content.
    """
    from langchain.document_loaders import UnstructuredWordDocumentLoader
    temp_path = f"temp_{filename}"
    with open(temp_path, "wb") as f:
        f.write(file_obj.getvalue())
    loader = UnstructuredWordDocumentLoader(temp_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = filename
    os.remove(temp_path)
    return docs

def load_excel(file_obj, filename):
    """
    Processes Excel files using pandas.
    """
    try:
        dfs = pd.read_excel(file_obj, sheet_name=None)
    except Exception as e:
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
        doc = Document(
            page_content=sheet_text,
            metadata={"source": filename, "sheet": sheet_name, "columns": headers}
        )
        docs.append(doc)
    return docs

def load_csv(file_obj, filename):
    """
    Processes CSV files using pandas.
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return []
    headers = list(df.columns)
    csv_text = "CSV Data:\n"
    for idx, row in df.iterrows():
        row_desc = ", ".join([f"{col}: {row[col]}" for col in headers])
        csv_text += row_desc + "\n"
    doc = Document(
        page_content=csv_text,
        metadata={"source": filename, "columns": headers}
    )
    return [doc]

# You can also include your text splitter configuration here
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)
