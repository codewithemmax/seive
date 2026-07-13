#!/usr/bin/env python3
"""Local eval harness — run from repo root:

    python eval/run_eval.py [--tasks path/to/tasks.json]

Reports per-category accuracy and total simulated token spend.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env for local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from pipeline import process_task

SAMPLE_TASKS = [
    {"id": "math_001", "category": "math", "input": "What is 17 * 43 + sqrt(144)?",
     "_expected": "743"},
    {"id": "math_002", "category": "math", "input": "Solve for x: 2x^2 - 8 = 0",
     "_expected": "2"},
    {"id": "code_gen_001", "category": "code generation",
     "input": "Write a Python function that returns the factorial of n using recursion.",
     "_expected": "def factorial"},
    {"id": "code_debug_001", "category": "code debugging",
     "input": "Fix this Python function:\ndef add(a, b):\n    return a - b",
     "_expected": "return a + b"},
    {"id": "logic_001", "category": "logic/deduction",
     "input": "All mammals are warm-blooded. Whales are mammals. Therefore, are whales warm-blooded? (A) Yes (B) No (C) Cannot determine",
     "_expected": "Yes"},
    {"id": "sentiment_001", "category": "sentiment analysis",
     "input": "This product is absolutely terrible. I want my money back.",
     "_expected": "Negative"},
    {"id": "ner_001", "category": "named entity recognition",
     "input": "Extract named entities from: 'Apple Inc. was founded by Steve Jobs in Cupertino, California.'",
     "_expected": "Apple"},
    {"id": "factual_001", "category": "factual QA",
     "input": "What is the capital of France?",
     "_expected": "Paris"},
    {"id": "summarization_001", "category": "summarization",
     "input": "Summarize in one sentence: The Amazon rainforest, often referred to as the 'lungs of the Earth', produces 20% of the world's oxygen and is home to 10% of all species on the planet. Deforestation threatens this critical ecosystem.",
     "_expected": "Amazon"},
]

SEP = "=" * 72


def _passes(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def run_eval(tasks_with_expected: list) -> None:
    results_by_category: dict[str, list[bool]] = {}
    total_tokens = 0
    total_tasks = len(tasks_with_expected)
    passed = 0

    print(f"\n{SEP}")
    print(f"{'ID':<20} {'Category':<22} {'Pass':<6} {'Tokens':<8} {'Source'}")
    print(SEP)

    for task in tasks_with_expected:
        expected = task.pop("_expected", None)
        t0 = time.time()
        result = process_task(task)
        elapsed = time.time() - t0

        answer = result.get("answer", "")
        tokens = result.get("tokens_used", 0)
        source = result.get("source", "?")
        category = task.get("category", "unknown")

        ok = _passes(answer, expected) if expected else None
        if ok:
            passed += 1
        total_tokens += tokens
        results_by_category.setdefault(category, []).append(bool(ok))

        status = "PASS" if ok else ("N/A " if ok is None else "FAIL")
        print(f"{task['id']:<20} {category:<22} {status:<6} {tokens:<8} {source}  [{elapsed:.1f}s]")

    print(SEP)
    print(f"\nOverall accuracy : {passed}/{total_tasks} ({100*passed/total_tasks:.1f}%)")
    print(f"Total tokens used: {total_tokens}")
    print("\nPer-category breakdown:")
    for cat, outcomes in sorted(results_by_category.items()):
        n = len(outcomes)
        p = sum(outcomes)
        print(f"  {cat:<25} {p}/{n}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Sieve local eval harness")
    parser.add_argument("--tasks", help="Path to tasks JSON (with _expected fields)")
    args = parser.parse_args()

    if args.tasks:
        with open(args.tasks) as f:
            tasks = json.load(f)
    else:
        tasks = [dict(t) for t in SAMPLE_TASKS]

    run_eval(tasks)


if __name__ == "__main__":
    main()
