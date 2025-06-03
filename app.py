from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
from io import BytesIO
from langchain.docstore.document import Document
import pandas as pd

from agents.unstructured_agent.document_loaders import (
    load_pdf, load_docx, load_excel, load_csv, text_splitter
)
from agents.unstructured_agent.vector_store import get_vector_store
from agents.unstructured_agent.agent import HybridQAChain
from agents.database_agent.agent import build_sql_agent_with_memory
from agents.pandas_agent.agent import build_pandas_agent_with_memory

app = FastAPI(title="Analytiq API")

# In-memory state (very basic) -----------------------------------------
vector_store = get_vector_store()
sql_agent = None
pandas_agent = None

# ----------------------------------------------------------------------
@app.post("/docs/upload")
async def upload_docs(files: List[UploadFile] = File(...)):
    """Upload documents into the vector store."""
    for file in files:
        data = await file.read()
        buf = BytesIO(data)
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            docs = load_pdf(buf, file.filename)
        elif ext == "docx":
            docs = load_docx(buf, file.filename)
        elif ext == "xlsx":
            docs = load_excel(buf, file.filename)
        elif ext == "csv":
            docs = load_csv(buf, file.filename)
        else:
            text = data.decode("utf-8", errors="ignore")
            docs = [Document(page_content=text, metadata={"source": file.filename})]
        chunks = text_splitter.split_documents(docs)
        vector_store.add_documents(chunks, batch_size=128)
    vector_store.persist()
    return {"status": "ok", "files": [f.filename for f in files]}


@app.post("/docs/chat")
async def chat_docs(question: str = Form(...),
                    temperature: float = Form(0.0),
                    top_k_vector: int = Form(10),
                    top_k_rerank: int = Form(3),
                    sheet_filter: str | None = Form(None)):
    chain = HybridQAChain(
        temperature=temperature,
        top_k_vector=top_k_vector,
        top_k_rerank=top_k_rerank,
        sheet_filter=sheet_filter or None,
    )
    result = chain.run(question)
    sources = [d.metadata.get("source", "") for d in result.get("source_documents", [])]
    return {"answer": result.get("answer"), "sources": sources}


@app.post("/sql/connect")
async def connect_db(conn_str: str = Form(...), temperature: float = Form(0.0)):
    global sql_agent
    sql_agent = build_sql_agent_with_memory(conn_str, temperature=temperature)
    return {"status": "connected"}


@app.post("/sql/query")
async def run_sql(query: str = Form(...)):
    if sql_agent is None:
        return JSONResponse({"error": "Connect first."}, status_code=400)
    try:
        res = sql_agent.run(query)
        return {"result": str(res)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/pandas/upload")
async def upload_tables(files: List[UploadFile] = File(...), temperature: float = Form(0.0)):
    dfs = {}
    for file in files:
        data = await file.read()
        buf = BytesIO(data)
        name = file.filename
        if name.lower().endswith(".csv"):
            df = pd.read_csv(buf)
            dfs[name] = {"Sheet1": df}
        else:
            sheets = pd.read_excel(buf, sheet_name=None)
            dfs[name] = sheets
    global pandas_agent
    pandas_agent = build_pandas_agent_with_memory(dfs, temperature=temperature)
    return {"status": "loaded", "files": list(dfs)}


@app.post("/pandas/query")
async def query_tables(question: str = Form(...)):
    if pandas_agent is None:
        return JSONResponse({"error": "Upload tables first."}, status_code=400)
    try:
        out = pandas_agent.run(question)
        return {"result": str(out)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

