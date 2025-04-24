# agents/unstructured_agent/agent.py
import os, re, json, cohere
from typing import List

import streamlit as st
from langchain.schema import Document
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import OpenAI
from langchain_community.retrievers import BM25Retriever
from agents.unstructured_agent.vector_store import get_vector_store       

# ------------------------------------------------------------------
# Keys & clients
# ------------------------------------------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY") or st.secrets.get("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not COHERE_API_KEY or not OPENAI_API_KEY:
    st.error("❌  Missing COHERE_API_KEY or OPENAI_API_KEY.")
    st.stop()

cohere_client = cohere.Client(COHERE_API_KEY)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "You are an AI assistant for an Institutional Research department. "
    "Answer factually, cite sources, and ask follow‑ups when uncertain."
)

# ---------------- helpers ----------------
_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?")

def _looks_numeric(text: str) -> bool:
    return bool(_NUM_RE.search(text))

def _verify_numeric(answer: str, docs: list[Document]) -> bool:
    """If the answer contains numbers, make sure at least one of them
    appears in the supporting docs."""
    if not _looks_numeric(answer):
        return True
    ans_nums = set(_NUM_RE.findall(answer))
    doc_nums = set(_NUM_RE.findall(" ".join(d.page_content for d in docs)))
    return bool(ans_nums & doc_nums)

# --------------- Cohere re‑rank ----------
def rerank_chunks(query: str,
                  docs: List[Document],
                  top_k: int = 3) -> List[Document]:
    if not docs:
        return []
    try:
        resp = cohere_client.rerank(
            model="rerank-3-nimble",
            query=query,
            documents=[d.page_content for d in docs],
            top_n=top_k,
        )
        return [docs[r.index] for r in resp.results]
    except Exception:
        return docs[:top_k]

# --------------- Hallucination check -----
def verify_answer(answer: str,
                  docs: List[Document],
                  llm) -> bool:
    context = "\n\n".join(d.page_content for d in docs[:5])
    prompt = (
        "If every factual statement in the ANSWER is supported by the PASSAGES, "
        "reply 'yes'; otherwise 'no'.\n\nANSWER:\n"
        f"{answer}\n\nPASSAGES:\n{context}\n"
    )
    return llm(prompt).strip().lower().startswith("yes")

# ================================================================
class HybridQAChain:
    """Hybrid (vector + BM25) retriever, Cohere re‑rank, numeric guard."""

    def __init__(
        self,
        temperature: float = 0.0,
        top_k_vector: int = 10,
        top_k_rerank: int = 3,
        sheet_filter: str | None = None,
    ):
        self.llm = OpenAI(temperature=temperature,
                          openai_api_key=OPENAI_API_KEY)

        self.mem = ConversationSummaryMemory(
            llm=self.llm,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
        )

        # Vector side (always initialized)
        self.vec_retriever = get_vector_store().as_retriever(
            search_kwargs={"k": top_k_vector, "filter": {"level": "row"}}
        )

        # BM25 initialized lazily
        self._kw_retriever: BM25Retriever | None = None

        self._weights = (0.6, 0.4)
        self.k_rerank = top_k_rerank
        self.sheet_filter = sheet_filter

    # ---------------------------------------------------------
    def _fetch(self, query: str) -> List[Document]:
        # ① semantic search
        vector_docs = self.vec_retriever.get_relevant_documents(query)

        # ② keyword search – build BM25 index the *first* time
        if self._kw_retriever is None and vector_docs:
            self._kw_retriever = BM25Retriever.from_documents(vector_docs)

        keyword_docs = (
            self._kw_retriever.get_relevant_documents(query)
            if self._kw_retriever else []
        )

        # Combine vector and keyword docs
        docs = vector_docs[: int(self._weights[0] * self.k_rerank)] + \
               keyword_docs[: int(self._weights[1] * self.k_rerank)]

        # Optional sheet restriction
        if self.sheet_filter:
            docs = [d for d in docs
                    if d.metadata.get("sheet_name") == self.sheet_filter]

        return rerank_chunks(query, docs, top_k=self.k_rerank)

    # ---------------------------------------------------------
    def run(self, query: str):
        docs = self._fetch(query)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = (
            f"{SYSTEM_PROMPT}\n\nUser question: {query}\n\n"
            f"Relevant context:\n{context}\n\n"
            "Answer the question. If not enough info, ask follow‑up or say 'I don't know'."
        )
        draft = self.llm(prompt)

        is_valid = (verify_answer(draft, docs, self.llm)
                    and _verify_numeric(draft, docs))

        answer = draft if is_valid else (
            "I'm not fully confident the retrieved info is sufficient. "
            "Could you provide more context?"
        )

        return {"answer": answer, "source_documents": docs}
