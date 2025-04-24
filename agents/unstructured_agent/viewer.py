# agent/unstructured_agent/viewer.py
import base64, streamlit as st, pandas as pd
from io import BytesIO

def _show_csv_or_excel(bytes_obj, ext, sample=500):
    if ext == "xlsx":
        df = pd.read_excel(BytesIO(bytes_obj), nrows=sample)
    else:
        df = pd.read_csv(BytesIO(bytes_obj), nrows=sample)
    st.markdown(f"Showing first {len(df)} rows (max {sample})")
    st.dataframe(df)

def view_document(filename, raw_files):
    ext = filename.split(".")[-1].lower()
    data = raw_files.get(filename)
    if not data:
        st.warning("File not found in session.")
        return

    if ext == "pdf":
        b64 = base64.b64encode(data).decode("utf-8")
        st.components.v1.html(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            'width="700" height="900"></iframe>', height=900
        )

    elif ext == "docx":
        import docx
        doc = docx.Document(BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
        st.text_area("DOCX preview (plain text):", text, height=500)

    elif ext in ("xlsx", "csv"):
        _show_csv_or_excel(data, ext)

    else:
        st.info("Unsupported preview type.")
