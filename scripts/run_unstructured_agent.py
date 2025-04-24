# scripts/run_unstructured_agent.py
from config.settings import OPENAI_API_KEY
from agents.unstructured_agent.agent import UnstructuredAgent

def main():
    agent = UnstructuredAgent(temperature=0.0, top_k=3)
    print("✅ Unstructured Agent built.")
    example = "Summarize the key points of the document 'sample.pdf'."
    # you’ll need to have already ingested 'sample.pdf' into your vector store
    result = agent.run(example)
    print("▶️", result)

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing!")
    main()
