# Dockerfile — System-3/0 Demo API
# Base: Ubuntu 22.04 + Python 3.11 + Ollama + models + uvicorn
# Target image size: <8 GB (Ollama + phi3:mini + llama3.1:8b ~5 GB)
#
# Build:  docker build -t mas-demo .
# Run:    docker run -p 8000:8000 mas-demo
# Compose: docker-compose up

FROM ubuntu:22.04

# ── system packages ──────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        curl \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ── Ollama ───────────────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.ai/install.sh | sh

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt ./
# Install only the server-relevant subset to keep the image lean.
# Exclude GUI deps (pynput, pywebview, pystray, Pillow) that aren't needed.
RUN pip install --no-cache-dir \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.27.0" \
        "pydantic[email]>=2.0" \
        "requests>=2.31.0" \
        "jsonschema>=4.21.0" \
        "langgraph>=0.2.0" \
        "langchain-core>=0.3.0" \
        "numpy>=1.26.0" \
        "scipy>=1.11.0" \
        "rich>=13.0.0" \
        "httpx>=0.27.0" \
        "anthropic>=0.25.0"

# ── application code ─────────────────────────────────────────────────────────
COPY . .
RUN pip install --no-cache-dir -e .

# ── persistent data directories ───────────────────────────────────────────────
RUN mkdir -p /app/data /tmp/mas-runs

# ── model pull script ─────────────────────────────────────────────────────────
# Pulls models at build time so the first request isn't slow.
# To skip (and pull at runtime), comment out the RUN line below.
# Note: this adds ~5 GB to the image. Use --build-arg SKIP_MODEL_PULL=1
# to skip during development builds.
ARG SKIP_MODEL_PULL=0
RUN if [ "$SKIP_MODEL_PULL" = "0" ]; then \
        ollama serve & \
        sleep 5 && \
        ollama pull phi3:mini && \
        ollama pull llama3.1:8b && \
        pkill ollama || true ; \
    fi

# ── entrypoint ───────────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV RUNS_DIR=/tmp/mas-runs
ENV OLLAMA_URL=http://localhost:11434
ENV API_VERSION=0.1.0

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
