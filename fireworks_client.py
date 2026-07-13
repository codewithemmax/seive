"""Layer 3 — Fireworks AI escalation.

All config from environment variables — nothing hardcoded.
Picks cheapest model per category, hard-caps tokens, uses terse prompts.
"""
import os
import logging
import json
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")

# Cheapest models per category — overridable via env
# deepseek-v4-pro: clean content field, low prompt tokens
# glm-5p1: fallback for longer outputs
_DEFAULT_MODELS = {
    "factual qa":      "accounts/fireworks/models/deepseek-v4-pro",
    "summarization":   "accounts/fireworks/models/deepseek-v4-pro",
    "logic/deduction": "accounts/fireworks/models/deepseek-v4-pro",
    "code debugging":  "accounts/fireworks/models/deepseek-v4-pro",
    "code generation": "accounts/fireworks/models/deepseek-v4-pro",
    "sentiment analysis": "accounts/fireworks/models/deepseek-v4-pro",
    "named entity recognition": "accounts/fireworks/models/deepseek-v4-pro",
    "default":         "accounts/fireworks/models/deepseek-v4-pro",
}

# Hard token caps per category — must be high enough for reasoning models to finish
_MAX_TOKENS = {
    "factual qa":      150,
    "summarization":   250,
    "logic/deduction": 150,
    "code debugging":  600,
    "code generation": 600,
    "sentiment analysis": 100,
    "named entity recognition": 250,
    "default":         150,
}

# Terse answer-only prompts — no chain-of-thought, minimal tokens
_PROMPTS = {
    "factual qa": "Answer in one sentence.\nQ: {input}\nA:",
    "summarization": "One sentence summary.\nText: {input}\nSummary:",
    "logic/deduction": "Answer letter only.\nQ: {input}\nA:",
    "code debugging": "Return corrected Python code only.\nCode:\n{input}\nFixed:",
    "code generation": "Return Python code only.\nRequest: {input}\nCode:",
    "sentiment analysis": "One word: Positive, Negative, or Neutral.\nText: {input}\nSentiment:",
    "named entity recognition": "List entities and types, one per line.\nText: {input}\nEntities:",
    "default": "Answer concisely.\nQ: {input}\nA:",
}


def _get_model(category: str) -> str:
    env_key = f"FIREWORKS_MODEL_{category.upper().replace(' ', '_').replace('/', '_')}"
    return os.environ.get(env_key) or _DEFAULT_MODELS.get(category.lower(), _DEFAULT_MODELS["default"])


def _extract_answer(text: str, category: str) -> str:
    """Clean up answer text — strip leading/trailing whitespace and markdown artifacts."""
    text = text.strip()
    # Strip markdown code fences if the category isn't code
    if category not in ("code debugging", "code generation"):
        import re
        text = re.sub(r"```[\w]*\n?", "", text).strip()
    return text


def _chat_completion(model: str, prompt: str, max_tokens: int) -> tuple[str, int]:
    """Returns (answer_text, tokens_used). Raises on HTTP error."""
    url = f"{FIREWORKS_BASE_URL.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant. Answer directly with no explanation, no reasoning steps, no preamble."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())

    text = (
        data["choices"][0]["message"].get("content")
        or data["choices"][0]["message"].get("reasoning_content")
        or ""
    ).strip()
    # Count only completion tokens (reasoning tokens are billed but we want accurate count)
    tokens = data.get("usage", {}).get("total_tokens", max_tokens)
    return text, tokens


def escalate(task: dict) -> dict:
    """Call Fireworks. Returns result dict with tokens_used > 0."""
    if not FIREWORKS_API_KEY:
        logger.error("FIREWORKS_API_KEY not set — cannot escalate")
        return {"answer": "", "confidence": 0.0, "tokens_used": 0, "source": "fireworks_no_key"}

    category = task.get("category", "default").lower().strip()
    prompt_template = _PROMPTS.get(category, _PROMPTS["default"])
    prompt = prompt_template.format(input=task.get("input", ""))
    model = _get_model(category)
    max_tok = _MAX_TOKENS.get(category, _MAX_TOKENS["default"])

    try:
        answer, tokens = _chat_completion(model, prompt, max_tok)
        answer = _extract_answer(answer, category)
        logger.info("Fireworks used %d tokens for task %s", tokens, task.get("id"))
        return {
            "answer": answer,
            "confidence": 0.9,
            "tokens_used": tokens,
            "source": f"fireworks:{model}",
        }
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        logger.error("Fireworks HTTP %d: %s", e.code, body)
        return {"answer": "", "confidence": 0.0, "tokens_used": 0, "source": f"fireworks_error:{e.code}"}
    except Exception as e:
        logger.error("Fireworks error: %s", e)
        return {"answer": "", "confidence": 0.0, "tokens_used": 0, "source": "fireworks_error"}
