"""Sieve — Streamlit Demo App for AMD Hackathon Act II pitch."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sieve — Token-Efficient AI Router",
    page_icon="🪣",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.layer-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
}
.l1 { background: #d4edda; color: #; }
.l2 { background: #cce5ff; color: #004085; }
.l3 { background: #fff3cd; color: #856404; }
.metric-box {
    background: #f8f9fa;Fix demo
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🪣 Sieve")
st.subheader("Hybrid Token-Efficient Routing Agent — AMD Hackathon Act II")
st.markdown(
    "Sieve solves tasks using the **cheapest layer possible**. "
    "Math, code, and logic are solved locally at **zero token cost**. "
    "Only hard tasks escalate to Fireworks AI."
)
st.divider()

# ── Sidebar — architecture explainer ─────────────────────────────────────────
with st.sidebar:
    st.header("How Sieve Works")
    st.markdown("""
**3-Layer Cascade:**

🟢 **Layer 1 — Deterministic (0 tokens)**
- Math → sympy solver
- Code → sandboxed subprocess
- Logic → syllogism parser

🔵 **Layer 2 — Local Model (0 tokens)**
- Qwen2.5-1.5B Q4 via llama-cpp
- Sentiment, NER, factual QA
- Confidence gate: 2 samples compared

🟡 **Layer 3 — Fireworks AI (billed)**
- Only when local confidence < 0.6
- Cheapest model per category
- Hard token caps enforced
""")
    st.divider()
    st.markdown("**Exact-match cache:** SHA-256 hashes every task. Duplicates cost zero tokens.")
    st.divider()
    api_key_input = st.text_input("Fireworks API Key", type="password",
                                   value=os.environ.get("FIREWORKS_API_KEY", ""),
                                   help="Required only for factual QA / summarization escalation")
    if api_key_input:
        os.environ["FIREWORKS_API_KEY"] = api_key_input

# ── Category presets ──────────────────────────────────────────────────────────
PRESETS = {
    "Math — expression":        ("math",                     "What is 17 * 43 + sqrt(144)?"),
    "Math — equation":          ("math",                     "Solve for x: 2x^2 - 8 = 0"),
    "Code generation":          ("code generation",          "Write a Python function that returns the factorial of n using recursion."),
    "Code debugging":           ("code debugging",           "Fix this Python function:\ndef add(a, b):\n    return a - b"),
    "Logic / deduction":        ("logic/deduction",          "All mammals are warm-blooded. Whales are mammals. Therefore, are whales warm-blooded? (A) Yes (B) No (C) Cannot determine"),
    "Sentiment analysis":       ("sentiment analysis",       "This product is absolutely terrible. I want my money back."),
    "Named entity recognition": ("named entity recognition", "Apple Inc. was founded by Steve Jobs in Cupertino, California."),
    "Factual QA":               ("factual QA",               "What is the capital of France?"),
    "Summarization":            ("summarization",            "The Amazon rainforest produces 20% of the world's oxygen and is home to 10% of all species on the planet. Deforestation threatens this critical ecosystem."),
    "Custom":                   ("", ""),
}

# ── Input panel ───────────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Task Input")

    preset = st.selectbox("Load a preset", list(PRESETS.keys()))
    preset_cat, preset_input = PRESETS[preset]

    category = st.selectbox(
        "Category",
        ["math", "code generation", "code debugging", "logic/deduction",
         "sentiment analysis", "named entity recognition", "factual QA", "summarization"],
        index=["math", "code generation", "code debugging", "logic/deduction",
               "sentiment analysis", "named entity recognition", "factual QA", "summarization"].index(preset_cat) if preset_cat else 0,
    )

    task_input = st.text_area("Task input", value=preset_input, height=160)
    task_id = st.text_input("Task ID (optional)", value="demo_001")

    run = st.button("Run Sieve", type="primary", use_container_width=True)

# ── Output panel ──────────────────────────────────────────────────────────────
with col_output:
    st.subheader("Result")

    if run:
        if not task_input.strip():
            st.warning("Enter a task input first.")
        else:
            task = {"id": task_id or "demo_001", "category": category, "input": task_input}

            with st.spinner("Running cascade..."):
                import logging
                logging.disable(logging.CRITICAL)
                try:
                    from pipeline import process_task
                    t0 = time.time()
                    result = process_task(task)
                    elapsed = time.time() - t0
                finally:
                    logging.disable(logging.NOTSET)

            answer = result.get("answer", "")
            confidence = result.get("confidence", 0.0)
            tokens = result.get("tokens_used", 0)
            source = result.get("source", "")

            # Layer badge
            if "math_solver" in source or "code_solver" in source or "logic_solver" in source:
                layer_label = "Layer 1 — Deterministic"
                badge_class = "l1"
                layer_emoji = "🟢"
            elif "local_model" in source:
                layer_label = "Layer 2 — Local Model"
                badge_class = "l2"
                layer_emoji = "🔵"
            elif "fireworks" in source:
                layer_label = "Layer 3 — Fireworks AI"
                badge_class = "l3"
                layer_emoji = "🟡"
            else:
                layer_label = "Fallback"
                badge_class = "l3"
                layer_emoji = "🔴"

            st.markdown(
                f'<div class="layer-badge {badge_class}">{layer_emoji} {layer_label}</div>',
                unsafe_allow_html=True,
            )

            # Answer
            if category in ("code generation", "code debugging"):
                st.code(answer.replace("```python", "").replace("```", "").strip(), language="python")
            else:
                st.success(answer if answer else "_(no answer)_")

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Tokens Used", tokens, delta="FREE" if tokens == 0 else None,
                      delta_color="normal" if tokens == 0 else "inverse")
            m2.metric("Confidence", f"{confidence:.0%}")
            m3.metric("Time", f"{elapsed:.2f}s")

            st.caption(f"Source: `{source}`")

            # Cache indicator
            if ":cached" in source:
                st.info("Cache hit — this exact task was seen before, zero additional cost.")
    else:
        st.info("Configure a task on the left and click **Run Sieve**.")

# ── Batch demo ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Batch Demo — Run All 9 Sample Tasks")
st.markdown("Runs the full sample suite and shows the token breakdown across all categories.")

if st.button("Run Batch", use_container_width=True):
    from pipeline import process_task

    BATCH = [
        {"id": "math_001",          "category": "math",                     "input": "What is 17 * 43 + sqrt(144)?"},
        {"id": "math_002",          "category": "math",                     "input": "Solve for x: 2x^2 - 8 = 0"},
        {"id": "code_gen_001",      "category": "code generation",          "input": "Write a Python function that returns the factorial of n using recursion."},
        {"id": "code_debug_001",    "category": "code debugging",           "input": "Fix this Python function:\ndef add(a, b):\n    return a - b"},
        {"id": "logic_001",         "category": "logic/deduction",          "input": "All mammals are warm-blooded. Whales are mammals. Therefore, are whales warm-blooded? (A) Yes (B) No (C) Cannot determine"},
        {"id": "sentiment_001",     "category": "sentiment analysis",       "input": "This product is absolutely terrible. I want my money back."},
        {"id": "ner_001",           "category": "named entity recognition", "input": "Apple Inc. was founded by Steve Jobs in Cupertino, California."},
        {"id": "factual_001",       "category": "factual QA",               "input": "What is the capital of France?"},
        {"id": "summarization_001", "category": "summarization",            "input": "The Amazon rainforest produces 20% of the world's oxygen and is home to 10% of all species. Deforestation threatens this ecosystem."},
    ]

    rows = []
    total_tokens = 0
    progress = st.progress(0, text="Running tasks...")

    import logging
    logging.disable(logging.CRITICAL)
    try:
        for i, task in enumerate(BATCH):
            t0 = time.time()
            result = process_task(task)
            elapsed = time.time() - t0
            tokens = result.get("tokens_used", 0)
            total_tokens += tokens
            source = result.get("source", "")

            if "math_solver" in source or "code_solver" in source or "logic_solver" in source:
                layer = "L1 Deterministic"
            elif "local_model" in source:
                layer = "L2 Local Model"
            elif "fireworks" in source:
                layer = "L3 Fireworks"
            else:
                layer = "Fallback"

            rows.append({
                "ID": task["id"],
                "Category": task["category"],
                "Layer": layer,
                "Tokens": tokens,
                "Confidence": f"{result.get('confidence', 0):.0%}",
                "Time": f"{elapsed:.1f}s",
                "Answer": result.get("answer", "")[:60],
            })
            progress.progress((i + 1) / len(BATCH), text=f"Completed {i+1}/{len(BATCH)}")
    finally:
        logging.disable(logging.NOTSET)

    progress.empty()

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary metrics
    free_tasks = sum(1 for r in rows if r["Tokens"] == 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tokens Used", total_tokens)
    c2.metric("Tasks Solved Free", f"{free_tasks}/{len(BATCH)}")
    c3.metric("Token Savings", f"{free_tasks/len(BATCH):.0%}")

    # Token breakdown chart
    import pandas as pd
    chart_data = pd.DataFrame({
        "Category": [r["Category"] for r in rows],
        "Tokens": [r["Tokens"] for r in rows],
    }).set_index("Category")
    st.bar_chart(chart_data)

st.divider()
st.caption("Sieve — AMD Developer Hackathon Act II · Track 1 · Built with Fireworks AI + llama-cpp-python + sympy")
