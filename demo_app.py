# demo_app.py
import streamlit as st
import json
from router import classify
from main import run_pipeline  # reuse your actual pipeline

st.title("Sieve — Zero-Token-First Routing Agent")
st.caption("AMD Developer Hackathon Act II · Track 1")

uploaded = st.file_uploader("Upload tasks.json", type="json")
sample = st.button("Or run built-in sample tasks")

if uploaded or sample:
    tasks = json.load(uploaded) if uploaded else json.load(open("eval/sample_tasks.json"))
    results, trace = run_pipeline(tasks, return_trace=True)
    
    total_tokens = sum(t["fireworks_tokens"] for t in trace)
    local_count = sum(1 for t in trace if t["layer"] != "fireworks")
    
    col1, col2 = st.columns(2)
    col1.metric("Solved locally", f"{local_count}/{len(tasks)}")
    col2.metric("Fireworks tokens spent", total_tokens)
    
    st.dataframe(trace)
    st.json(results)