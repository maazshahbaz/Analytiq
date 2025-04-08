# chat.py

from langchain.schema import HumanMessage, AIMessage
# Previously we used ConversationalRetrievalChain with ConversationBufferMemory.
# Now we use ConversationSummaryMemory to automatically summarize long histories.
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory  # updated memory type
from langchain_community.llms import OpenAI
from vector_store import get_vector_store
from langchain.prompts import PromptTemplate
import os

def convert_history(history):
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

# ----------------------
# Domain-Specific System Prompt
# ----------------------
SYSTEM_PROMPT = (
    "You are an AI assistant for an Institutional Research department. "
    "Your goal is to provide factual answers, reference relevant document sources, "
    "and maintain a professional and analytical tone."
)

# Build a custom QA prompt template that embeds the system instructions.
QA_PROMPT = PromptTemplate.from_template(
    SYSTEM_PROMPT
    + "\n\nUser question: {question}\n\nContext:\n{context}\n\nAnswer:"
)

def create_qa_chain(chat_history):
    llm = OpenAI(temperature=0)
    # Use ConversationSummaryMemory to summarize older parts of the conversation.
    memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
    if chat_history:
        memory.chat_memory.messages = convert_history(chat_history)
    vector_store = get_vector_store()
    # Create the QA chain with our custom QA prompt, which includes our system prompt.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
        qa_prompt=QA_PROMPT  # NEW: this injects our system/context instructions into every query.
    )
    return qa_chain
