import os, re
from typing import List
import streamlit as st
import cohere
from langchain.schema import Document
from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import OpenAI
from vector_store import filtered_search

cohere_client = cohere.Client(st.secrets["COHERE_API_KEY"])
SYSTEM_PROMPT = (
    "You are an AI assistant for an Institutional Research department. "
    "Answer factually, cite sources, ask follow‑ups when uncertain."
)

# ---------- Reranking -------------------------------------------------
def rerank_chunks(query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
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

# ---------- Hallucination check --------------------------------------
def verify_answer(answer: str, docs: List[Document], llm) -> bool:
    context = "\n\n".join(d.page_content for d in docs[:5])
    prompt = (
        "If every factual statement in the ANSWER is supported by the PASSAGES, "
        "reply 'yes'; otherwise 'no'.\n\nANSWER:\n"
        f"{answer}\n\nPASSAGES:\n{context}\n"
    )
    return llm(prompt).strip().lower().startswith("yes")

# ---------- Main Chain -----------------------------------------------
class HybridQAChain:
    def __init__(
        self,
        temperature: float = 0.0,
        top_k_vector: int = 10,
        top_k_rerank: int = 3,
        sheet_filter: str | None = None,
    ):
        self.llm = OpenAI(temperature=temperature)
        self.mem = ConversationSummaryMemory(
            llm=self.llm, return_messages=True, memory_key="chat_history", output_key="answer"
        )
        self.k_vec = top_k_vector
        self.k_rerank = top_k_rerank
        self.sheet_filter = sheet_filter

    def run(self, query: str):
        # 1) metadata filter – focus on row‑level chunks, optional sheet restriction
        meta = {"level": "row"}
        if self.sheet_filter:
            meta["sheet_name"] = self.sheet_filter

        initial = filtered_search(query, k=self.k_vec, where=meta)
        final_docs = rerank_chunks(query, initial, top_k=self.k_rerank)

        context = "\n\n".join(d.page_content for d in final_docs)
        prompt = (
            f"{SYSTEM_PROMPT}\n\nUser question: {query}\n\n"
            f"Relevant context:\n{context}\n\n"
            "Answer the question. If not enough info, ask follow‑up or say 'I don't know'."
        )
        draft = self.llm(prompt)
        if verify_answer(draft, final_docs, self.llm):
            answer = draft
        else:
            answer = ("I'm not fully confident the retrieved info is sufficient. "
                      "Could you provide more context?")
        return {"answer": answer, "source_documents": final_docs}
