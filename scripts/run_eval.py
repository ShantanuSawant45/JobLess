import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from app.retrieval.search import search
from app.generation.prompts import build_prompt
from app.generation.llm import generate

EVAL_DIR = Path("eval")
RESULTS_DIR = EVAL_DIR / "results"
QUESTIONS_FILE = EVAL_DIR / "questions.json"


def evaluate_recall(context_text: str, question: str, expected_answer: str) -> bool:
    time.sleep(4)  # Avoid rate limits
    prompt = f"""You are an expert evaluator.
Does the following context contain the information needed to answer the question with the expected answer?

Context:
{context_text}

Question: {question}
Expected Answer: {expected_answer}

Reply with only YES or NO.
"""
    try:
        response = generate(prompt).strip().upper()
        return "YES" in response
    except Exception as e:
        print(f"Error evaluating recall: {e}")
        return False


def evaluate_faithfulness(context_text: str, question: str, answer: str) -> bool:
    time.sleep(4)  # Avoid rate limits
    prompt = f"""You are an expert evaluator.
Is the following answer strictly based ONLY on the provided context, without bringing in outside knowledge?

Context:
{context_text}

Question: {question}
Answer: {answer}

Reply with only YES or NO.
"""
    try:
        response = generate(prompt).strip().upper()
        return "YES" in response
    except Exception as e:
        print(f"Error evaluating faithfulness: {e}")
        return False


def run_eval():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(QUESTIONS_FILE, "r") as f:
        questions = json.load(f)

    results = []
    recall_correct = 0
    faithfulness_correct = 0

    print(f"Starting evaluation of {len(questions)} questions...")
    
    for idx, q_data in enumerate(questions):
        company = q_data["company"]
        question = q_data["question"]
        expected_answer = q_data["expected_answer"]

        print(f"\n[{idx+1}/{len(questions)}] Q: {question} (Company: {company})")

        # 1. Retrieve chunks
        try:
            chunks = search(question=question, company=company, top_k=5)
        except Exception as e:
            print(f"Retrieval failed: {e}")
            chunks = []

        context_text = "\n\n".join([c.get("text", "") for c in chunks])

        # 2. Generate answer
        if chunks:
            prompt = build_prompt(question=question, chunks=chunks)
            try:
                time.sleep(4)  # Avoid rate limits
                answer = generate(prompt)
            except Exception as e:
                print(f"Generation failed: {e}")
                answer = "Error generating answer."
        else:
            answer = "No chunks retrieved."

        # 3. Evaluate Recall
        recall_ok = evaluate_recall(context_text, question, expected_answer) if chunks else False
        if recall_ok:
            recall_correct += 1

        # 4. Evaluate Faithfulness
        faithfulness_ok = evaluate_faithfulness(context_text, question, answer) if chunks else False
        if faithfulness_ok:
            faithfulness_correct += 1

        print(f"  Recall: {'PASS' if recall_ok else 'FAIL'}")
        print(f"  Faithfulness: {'PASS' if faithfulness_ok else 'FAIL'}")

        results.append({
            "company": company,
            "question": question,
            "expected_answer": expected_answer,
            "retrieved_chunks": [c.get("text", "") for c in chunks],
            "generated_answer": answer,
            "recall": recall_ok,
            "faithfulness": faithfulness_ok
        })
        
        # Add a short sleep to avoid rate limits on LLM API
        time.sleep(4)

    total = len(questions)
    recall_rate = recall_correct / total if total > 0 else 0
    faithfulness_rate = faithfulness_correct / total if total > 0 else 0

    print("\n==============================")
    print("EVALUATION RESULTS")
    print("==============================")
    print(f"Total Questions: {total}")
    print(f"Recall@5: {recall_rate:.2%} ({recall_correct}/{total})")
    print(f"Faithfulness: {faithfulness_rate:.2%} ({faithfulness_correct}/{total})")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "metrics": {
            "recall@5": recall_rate,
            "faithfulness": faithfulness_rate
        },
        "details": results
    }

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"run_{timestamp_str}.json"
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    run_eval()
