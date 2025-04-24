# scripts/eval_unstructured.py
import json
from pathlib import Path

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from agents.unstructured_agent.agent import HybridQAChain

def main():
    # 1) Initialize the RAG chain
    chain = HybridQAChain(temperature=0.0, top_k_vector=10, top_k_rerank=3)
    print("✅ HybridQAChain ready for evaluation.\n")

    # 2) Load gold questions from scripts/gold.jsonl
    gold_path = Path(__file__).parent / "gold.jsonl"
    if not gold_path.exists():
        raise FileNotFoundError(f"{gold_path} not found.")

    gold_qs, gen_ans, ctxs = [], [], []
    with open(gold_path, "r") as fh:
        for line in fh:
            rec = json.loads(line)
            gold_qs.append(rec["question"])
            out = chain.run(rec["question"])
            gen_ans.append(out["answer"])
            ctxs.append([d.page_content for d in out["source_documents"]])

    # 3) Compute RAG metrics
    results_df = evaluate(
        questions=gold_qs,
        answers=gen_ans,
        contexts=ctxs,
        metrics=[faithfulness, answer_relevancy, context_precision],
    ).to_pandas()

    # 4) Display
    print("=== RAG Evaluation Results ===")
    print(results_df)

if __name__ == "__main__":
    main()
