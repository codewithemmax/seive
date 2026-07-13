"""Layer 2 — local GGUF model via llama-cpp-python.

Handles: sentiment, NER, factual QA, summarization, logic (fallback), code (fallback).
Confidence gate: run 2 samples; if they agree → high confidence, else → low (escalate).
"""
import os
import re
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH",
    "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
)

_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    try:
        from llama_cpp import Llama
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=int(os.environ.get("LLM_THREADS", "4")),
            verbose=False,
        )
        logger.info("Local model loaded from %s", MODEL_PATH)
    except Exception as e:
        logger.warning("Local model unavailable: %s", e)
        _llm = None
    return _llm


_PROMPTS = {
    "sentiment analysis": (
        "Classify the sentiment of the following text as exactly one word: "
        "Positive, Negative, or Neutral.\nText: {input}\nSentiment:"
    ),
    "named entity recognition": (
        "Extract named entities from the text below. "
        "List each entity and its type (PERSON, ORG, LOC, DATE, etc.) on a new line.\n"
        "Text: {input}\nEntities:"
    ),
    "factual QA": (
        "Answer the following question in one short sentence.\nQuestion: {input}\nAnswer:"
    ),
    "summarization": (
        "Summarize the following text in one concise sentence.\nText: {input}\nSummary:"
    ),
    "logic/deduction": (
        "Answer the following logic question. Give only the answer letter or a brief answer.\n"
        "Question: {input}\nAnswer:"
    ),
    "code debugging": (
        "Fix the bug in the following Python code. Return only the corrected code.\n"
        "Code:\n{input}\nFixed code:"
    ),
    "code generation": (
        "Write a Python function as requested. Return only the code.\nRequest: {input}\nCode:"
    ),
}

_MAX_TOKENS = {
    "sentiment analysis": 8,
    "named entity recognition": 120,
    "factual QA": 60,
    "summarization": 80,
    "logic/deduction": 20,
    "code debugging": 300,
    "code generation": 300,
}


def _sample(llm, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False,
        stop=["\n\n", "Question:", "Text:"],
    )
    return out["choices"][0]["text"].strip()


def _answers_agree(a: str, b: str, category: str) -> bool:
    """Heuristic agreement check."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    if category == "sentiment analysis":
        # Both must contain same sentiment word
        for word in ("positive", "negative", "neutral"):
            if word in a and word in b:
                return True
        return False
    # For other categories: agree if first 30 chars match or one contains the other
    return a[:30] == b[:30] or a in b or b in a


def run_local(task: dict) -> dict:
    """Returns result dict. confidence < 0.6 signals escalation needed."""
    llm = _get_llm()
    if llm is None:
        return {"answer": "", "confidence": 0.0, "tokens_used": 0, "source": "local_model_unavailable"}

    category = task.get("category", "factual QA")
    prompt_template = _PROMPTS.get(category, _PROMPTS["factual QA"])
    prompt = prompt_template.format(input=task.get("input", ""))
    max_tok = _MAX_TOKENS.get(category, 80)

    try:
        ans1 = _sample(llm, prompt, max_tok, temperature=0.0)
        ans2 = _sample(llm, prompt, max_tok, temperature=0.3)

        agree = _answers_agree(ans1, ans2, category)
        confidence = 0.85 if agree else 0.45

        # Use deterministic answer (temp=0) as primary
        answer = ans1

        # Post-process sentiment to single word
        if category == "sentiment analysis":
            for word in ("positive", "negative", "neutral"):
                if word in answer.lower():
                    answer = word.capitalize()
                    break

        tokens_used = 0  # local = free
        return {
            "answer": answer,
            "confidence": confidence,
            "tokens_used": tokens_used,
            "source": "local_model",
        }
    except Exception as e:
        logger.error("Local model inference error: %s", e)
        return {"answer": "", "confidence": 0.0, "tokens_used": 0, "source": "local_model_error"}
