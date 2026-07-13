"""Pipeline orchestrator — 3-layer cascade with cache, timeout, and fail-safe."""
import logging
import signal
import os
from contextlib import contextmanager

import cache
from solvers import solve_math, solve_code_debug, solve_code_gen, solve_logic
from local_model import run_local
from fireworks_client import escalate

logger = logging.getLogger(__name__)

TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", "25"))  # seconds
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))

# Categories handled by Layer 1 deterministic solvers
_L1_DISPATCH = {
    "math": solve_math,
    "code debugging": solve_code_debug,
    "code generation": solve_code_gen,
    "logic/deduction": solve_logic,
}

# Categories that go straight to Layer 2 (local model), skip L1
_L2_DIRECT = {"sentiment analysis", "named entity recognition"}

# Categories where L2 → L3 escalation is allowed (sentiment/NER stay local)
_ESCALATE_CATEGORIES = {"factual qa", "summarization", "logic/deduction", "code debugging", "code generation"}


@contextmanager
def _timeout(seconds: int):
    """SIGALRM-based timeout (Unix only). On Windows, falls through."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Task exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _fallback(task: dict, reason: str) -> dict:
    return {
        "id": task.get("id", "unknown"),
        "answer": "",
        "confidence": 0.0,
        "tokens_used": 0,
        "source": f"fallback:{reason}",
    }


def process_task(task: dict) -> dict:
    """Run the 3-layer cascade for a single task. Never raises."""
    task_id = task.get("id", "unknown")

    # Cache check
    cached = cache.get(task)
    if cached:
        logger.info("Cache hit for task %s", task_id)
        return {**cached, "id": task_id, "source": cached.get("source", "") + ":cached"}

    try:
        with _timeout(TASK_TIMEOUT):
            result = _run_cascade(task)
    except TimeoutError:
        logger.warning("Task %s timed out", task_id)
        result = _fallback(task, "timeout")
    except Exception as e:
        logger.error("Unexpected error on task %s: %s", task_id, e)
        result = _fallback(task, str(e))

    result["id"] = task_id

    # Cache successful results
    if result.get("confidence", 0) > 0.5:
        cache.set(task, result)

    return result


def _run_cascade(task: dict) -> dict:
    category = task.get("category", "").lower().strip()

    # ── Layer 1: Deterministic solvers ───────────────────────────────────────
    l1_solver = _L1_DISPATCH.get(category)
    if l1_solver:
        try:
            result = l1_solver(task)
            if result:
                logger.info("L1 solved task (category=%s)", category)
                return result
        except Exception as e:
            logger.warning("L1 solver error (category=%s): %s", category, e)

    # ── Layer 2: Local model ──────────────────────────────────────────────────
    try:
        local_result = run_local(task)
    except Exception as e:
        logger.warning("Local model error: %s", e)
        local_result = {"answer": "", "confidence": 0.0, "tokens_used": 0, "source": "local_error"}

    confidence = local_result.get("confidence", 0.0)

    # High confidence or category doesn't support escalation → return local result
    if confidence >= CONFIDENCE_THRESHOLD or category not in _ESCALATE_CATEGORIES:
        if local_result.get("answer"):
            logger.info("L2 solved task (category=%s, conf=%.2f)", category, confidence)
            return local_result

    # ── Layer 3: Fireworks escalation ─────────────────────────────────────────
    logger.info("Escalating to Fireworks (category=%s, conf=%.2f)", category, confidence)
    fw_result = escalate(task)

    # If Fireworks also fails, fall back to whatever local gave us
    if not fw_result.get("answer") and local_result.get("answer"):
        logger.warning("Fireworks failed, using local answer as fallback")
        return local_result

    return fw_result
