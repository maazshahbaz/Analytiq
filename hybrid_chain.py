import re
from typing import List

from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from vector_store import get_vector_store

# ------------------------------------------------------------------------
# 1. Simple keyword filter (example)
# ------------------------------------------------------------------------
def keyword_filter(query: str, docs: List[Document]) -> List[Document]:
    """
    A naive keyword filter that keeps only docs containing any word from the user query.
    Example: splits user query by whitespace, does case-insensitive search.
    """
    query_words = set(re.findall(r"\w+", query.lower()))
    filtered_docs = []
    for doc in docs:
        text_lower = doc.page_content.lower()
        if any(word in text_lower for word in query_words):
            filtered_docs.append(doc)
    return filtered_docs

# ------------------------------------------------------------------------
# 2. Rerank chunks with an LLM prompt
# ------------------------------------------------------------------------
def rerank_chunks(query: str, docs: List[Document], llm=None, top_k: int = 3) -> List[Document]:
    """
    Uses an LLM to rerank docs based on relevance to the query.
    Returns the top_k most relevant docs according to the LLM.
    """
    if llm is None:
        llm = OpenAI(temperature=0)

    # Build a prompt that shows each chunk and asks the LLM to rank them
    # by relevance to the query.
    prompt_text = f"You are a helpful assistant. A user asked: '{query}'. " \
                  f"Here are {len(docs)} document chunks. " \
                  "Rank them from most relevant (1) to least relevant. " \
                  "Return the final answer as a sorted list of indices.\n\n"

    for i, doc in enumerate(docs):
        snippet = doc.page_content[:200].replace("\n", " ")
        prompt_text += f"Document {i+1}: {snippet}\n\n"

    prompt_text += "Provide the ranking (1-based indices) in a comma-separated list. Example: '1,2,3,...'\n"

    # Call LLM
    response = llm(prompt_text)
    # Attempt to parse the output
    # e.g. "1,3,2"
    try:
        ranking_str = response.strip()
        # Convert "1,3,2" -> [0, 2, 1]
        ranking = [int(x.strip()) - 1 for x in ranking_str.split(",")]
    except:
        # Fallback: just pick them in order
        ranking = list(range(len(docs)))

    # Deduplicate indices and filter out invalid ones
    valid_indices = []
    for idx in ranking:
        if 0 <= idx < len(docs) and idx not in valid_indices:
            valid_indices.append(idx)

    # Return the top_k docs in the specified ranking order
    top_docs = [docs[i] for i in valid_indices[:top_k]]
    return top_docs

# ------------------------------------------------------------------------
# 3. Hybrid Search (Vector + Keyword)
# ------------------------------------------------------------------------
def hybrid_search(query: str, k: int = 10) -> List[Document]:
    """
    Performs a vector similarity search for the query, then applies a naive keyword filter.
    Returns up to k docs (before reranking).
    """
    vector_store = get_vector_store()

    # Step 1: Vector-based search
    # NOTE: If you're using LangChain's Chroma wrapper:
    #   results = vector_store.similarity_search(query, k=k)
    # or if you'd like raw Document objects:
    results = vector_store.similarity_search(query, k=k)

    # Step 2: Simple keyword filter
    # (If you'd prefer advanced BM25, you'd call a separate library or function)
    filtered = keyword_filter(query, results)

    # If the filter is too strict and returns zero, fallback to the raw vector results
    if not filtered:
        filtered = results

    return filtered

# ------------------------------------------------------------------------
# 4. HybridQAChain: A custom chain that uses hybrid_search + rerank_chunks
# ------------------------------------------------------------------------
class HybridQAChain:
    def __init__(self, temperature=0, top_k_vector=10, top_k_rerank=3):
        """
        :param temperature: LLM temperature
        :param top_k_vector: how many docs to fetch in the vector step
        :param top_k_rerank: how many docs to keep after reranking
        """
        self.llm = OpenAI(temperature=temperature)
        self.top_k_vector = top_k_vector
        self.top_k_rerank = top_k_rerank
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def run(self, query: str) -> dict:
        """
        Returns a dictionary with keys: {"answer": str, "source_documents": List[Document]}
        """
        # Step 1: Hybrid search (vector + keyword)
        initial_docs = hybrid_search(query, k=self.top_k_vector)

        # Step 2: LLM-based reranking
        final_docs = rerank_chunks(query, initial_docs, llm=self.llm, top_k=self.top_k_rerank)

        # Step 3: Combine the final_docs into a single prompt to get the final answer
        # We'll do something simple: feed these docs and the query to the LLM
        context_text = "\n\n".join([doc.page_content for doc in final_docs])
        prompt_text = (
            f"User question: {query}\n\n"
            f"Relevant context from top documents:\n{context_text}\n\n"
            "Please provide a helpful and accurate answer, using the context if relevant."
        )
        final_answer = self.llm(prompt_text)

        return {
            "answer": final_answer,
            "source_documents": final_docs
        }
