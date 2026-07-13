# Sieve — Hybrid Token-Efficient Routing Agent

AMD Developer Hackathon Act II · Track 1

## What it is

Sieve is a containerized AI agent that processes tasks from `/input/tasks.json` and writes answers to `/output/results.json`. It minimizes Fireworks AI token spend by solving as much as possible locally before escalating.

## How it works

Three-layer cascade, evaluated in order per task:

| Layer | What runs | Token cost |
|-------|-----------|------------|
| 1 — Deterministic solvers | sympy (math), sandboxed subprocess (code), structural parser (logic) | 0 |
| 2 — Local GGUF model | Qwen2.5-1.5B-Instruct Q4_K_M via llama-cpp-python | 0 |
| 3 — Fireworks escalation | Cheapest allowed model, hard-capped tokens | Billed |

**Routing logic:**
- Math → Layer 1 (sympy). Falls through to Layer 2 only if sympy can't parse it.
- Code debug/gen → Layer 1 (template + subprocess verify). Falls through to Layer 3 if no template matches.
- Logic/deduction → Layer 1 (syllogism parser). Falls through to Layer 2 → Layer 3.
- Sentiment / NER → Layer 2 directly (local model, no escalation).
- Factual QA / Summarization → Layer 2, escalate to Layer 3 only if confidence < 0.6.

**Confidence gate:** Layer 2 runs two samples (temperature 0 and 0.3). If they agree → confidence 0.85 (keep). If they disagree → confidence 0.45 (escalate).

**Exact-match cache:** Task inputs are SHA-256 hashed. Duplicate tasks reuse the cached answer at zero additional cost.

**Fail-safe:** Every task always produces an output entry. Timeouts and errors produce a low-confidence empty answer rather than crashing the run.

## Supported task categories

`math` · `code debugging` · `code generation` · `logic/deduction` · `sentiment analysis` · `named entity recognition` · `factual QA` · `summarization`

## Build the image

```bash
docker build -t sieve:latest .
```

The build downloads the Qwen2.5-1.5B-Instruct Q4_K_M GGUF (~1 GB) from Hugging Face. Ensure you have a stable internet connection and ~2 GB free disk space.

To use a pre-downloaded model:

```bash
docker build \
  --build-arg MODEL_URL="file:///models/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
  -t sieve:latest .
```

## Run locally

```bash
# Prepare input
mkdir -p input output
cp path/to/your/tasks.json input/tasks.json

# Run
docker run --rm \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  -e FIREWORKS_API_KEY=your_key_here \
  sieve:latest

# Results
cat output/results.json
```

## Run the eval harness (no Docker)

```bash
pip install -r requirements.txt

# Against built-in sample tasks
python eval/run_eval.py

# Against your own tasks (must include _expected fields for accuracy scoring)
python eval/run_eval.py --tasks eval/my_tasks.json
```

Sample output:
```
────────────────────────────────────────────────────────────
ID                   Category               Pass   Tokens   Source
────────────────────────────────────────────────────────────
math_001             math                   ✓      0        math_solver  [0.1s]
code_gen_001         code generation        ✓      0        code_solver  [0.2s]
factual_001          factual QA             ✓      0        local_model  [1.4s]
...
────────────────────────────────────────────────────────────
Overall accuracy : 8/9 (88.9%)
Total tokens used: 120
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FIREWORKS_API_KEY` | Yes | — | Fireworks AI API key |
| `FIREWORKS_BASE_URL` | No | `https://api.fireworks.ai/inference/v1` | Fireworks base URL |
| `LOCAL_MODEL_PATH` | No | `/models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | Path to GGUF model file |
| `INPUT_PATH` | No | `/input/tasks.json` | Input tasks file |
| `OUTPUT_PATH` | No | `/output/results.json` | Output results file |
| `TASK_TIMEOUT` | No | `25` | Per-task timeout in seconds |
| `CONFIDENCE_THRESHOLD` | No | `0.6` | Min confidence to skip Fireworks escalation |
| `LLM_THREADS` | No | `4` | CPU threads for local model inference |
| `CACHE_FILE` | No | `/tmp/sieve_cache.json` | Path for the exact-match cache |
| `FIREWORKS_MODEL_FACTUAL_QA` | No | llama4-scout-instruct-basic | Override model for factual QA |
| `FIREWORKS_MODEL_SUMMARIZATION` | No | llama4-scout-instruct-basic | Override model for summarization |
| `FIREWORKS_MODEL_CODE_DEBUGGING` | No | llama4-maverick-instruct-basic | Override model for code tasks |
| `FIREWORKS_MODEL_CODE_GENERATION` | No | llama4-maverick-instruct-basic | Override model for code tasks |

## Input / Output format

**Input** (`/input/tasks.json`):
```json
[
  { "id": "task_001", "category": "math", "input": "What is 12 * 8?" },
  { "id": "task_002", "category": "factual QA", "input": "Who wrote Hamlet?" }
]
```

**Output** (`/output/results.json`):
```json
[
  { "id": "task_001", "answer": "96", "confidence": 1.0, "tokens_used": 0, "source": "math_solver" },
  { "id": "task_002", "answer": "William Shakespeare", "confidence": 0.85, "tokens_used": 0, "source": "local_model" }
]
```
