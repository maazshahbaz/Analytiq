# viewer.py
import base64
import streamlit as st
import pandas as pd
from io import BytesIO

def view_document(filename, raw_files):
    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        pdf_bytes = raw_files.get(filename)
        if pdf_bytes:
            b64 = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="900" type="application/pdf"></iframe>'
            st.components.v1.html(pdf_display, height=900)
    elif ext == "docx":
        file_bytes = raw_files.get(filename)
        if file_bytes:
            st.text_area("Document Content", "Preview not available for DOCX.", height=500)
    elif ext in ["xlsx", "csv"]:
        file_bytes = raw_files.get(filename)
        if file_bytes:
            if ext == "xlsx":
                df = pd.read_excel(BytesIO(file_bytes))
            else:
                df = pd.read_csv(BytesIO(file_bytes))
            st.dataframe(df)
    else:
        st.write("Unsupported file type for viewer.")
