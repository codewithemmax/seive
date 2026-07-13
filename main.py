"""Sieve — Hybrid Token-Efficient Routing Agent.

Reads  /input/tasks.json
Writes /output/results.json
"""
import json
import logging
import os
import sys

# Load .env for local development (no-op if python-dotenv not installed or no .env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("sieve")

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")


def main():
    # ── Load tasks ────────────────────────────────────────────────────────────
    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)
    except Exception as e:
        logger.critical("Cannot read %s: %s", INPUT_PATH, e)
        sys.exit(1)

    if not isinstance(tasks, list):
        tasks = [tasks]

    logger.info("Loaded %d tasks from %s", len(tasks), INPUT_PATH)

    # Import here so model loading happens after startup log
    from pipeline import process_task

    results = []
    total_tokens = 0

    for task in tasks:
        task_id = task.get("id", f"task_{len(results)}")
        logger.info("Processing task %s (category=%s)", task_id, task.get("category"))
        result = process_task(task)
        results.append(result)
        total_tokens += result.get("tokens_used", 0)
        logger.info(
            "Task %s → source=%s conf=%.2f tokens=%d",
            task_id,
            result.get("source", "?"),
            result.get("confidence", 0.0),
            result.get("tokens_used", 0),
        )

    # ── Write results ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "Done. %d tasks processed. Total Fireworks tokens: %d. Results → %s",
        len(results), total_tokens, OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
