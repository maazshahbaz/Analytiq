"""
Evaluation harness for your RAG pipeline.

• Expects a file `gold.jsonl` in the project root where each line is:
  {"question": "...", "answer": "ground‑truth answer"}

• Runs the current HybridQAChain on every question
  and scores the outputs with RAGAS metrics.

Run:  python eval_harness.py
"""

import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from hybrid_chain import HybridQAChain

chain = HybridQAChain()

gold_qs, generated_ans, retrieved_ctxs = [], [], []
with open("gold.jsonl") as fh:
    for line in fh:
        rec = json.loads(line)
        gold_qs.append(rec["question"])
        res = chain.run(rec["question"])
        generated_ans.append(res["answer"])
        retrieved_ctxs.append([d.page_content for d in res["source_documents"]])

score_df = evaluate(
    questions=gold_qs,
    answers=generated_ans,
    contexts=retrieved_ctxs,
    metrics=[faithfulness, answer_relevancy, context_precision],
).to_pandas()

print("\n=== RAG evaluation ===")
print(score_df)
