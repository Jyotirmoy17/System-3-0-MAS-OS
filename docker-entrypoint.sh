#!/usr/bin/env bash
# docker-entrypoint.sh
# Start Ollama daemon in the background, then launch the FastAPI server.
set -euo pipefail

echo "[entrypoint] Starting Ollama daemon..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready (up to 30 s)
WAIT=0
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    if [ "$WAIT" -ge 30 ]; then
        echo "[entrypoint] WARNING: Ollama did not become ready in 30s — continuing anyway."
        break
    fi
    sleep 1
    WAIT=$((WAIT + 1))
done
echo "[entrypoint] Ollama ready (waited ${WAIT}s)."

# Pull models if not already available (handles the SKIP_MODEL_PULL=1 build case)
for MODEL in phi3:mini llama3.1:8b; do
    if ! ollama list 2>/dev/null | grep -q "^${MODEL%%:*}"; then
        echo "[entrypoint] Pulling $MODEL ..."
        ollama pull "$MODEL" || echo "[entrypoint] WARNING: could not pull $MODEL"
    fi
done

echo "[entrypoint] Starting uvicorn on port 8000..."
exec uvicorn mas.web.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
