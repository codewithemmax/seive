# ── Stage 1: build llama-cpp-python (CPU) ────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Build llama-cpp-python with CPU backend only
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_CUBLAS=OFF" \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

# ── Download model weights ────────────────────────────────────────────────────
# Qwen2.5-1.5B-Instruct Q4_K_M (~1.0 GB) from Hugging Face
ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
ARG MODEL_PATH="/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

RUN mkdir -p /models && \
    curl -L --retry 3 --retry-delay 5 -o "${MODEL_PATH}" "${MODEL_URL}"

# ── Runtime config ────────────────────────────────────────────────────────────
ENV LOCAL_MODEL_PATH="/models/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    INPUT_PATH="/input/tasks.json" \
    OUTPUT_PATH="/output/results.json" \
    LLM_THREADS="4" \
    TASK_TIMEOUT="25" \
    CONFIDENCE_THRESHOLD="0.6" \
    PYTHONUNBUFFERED="1"

# Volumes for I/O
VOLUME ["/input", "/output"]

CMD ["python", "main.py"]
