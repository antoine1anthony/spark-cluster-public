#!/bin/bash
# Brain LLM Entrypoint
# Starts both the prompt registry and vLLM server

set -e

echo "=== Brain LLM Startup ==="
echo "Starting Prompt Registry on port ${PROMPT_PORT:-8010}..."

# Start the prompt loader in the background
python3 /app/prompt_loader.py &
PROMPT_PID=$!

# Give prompt loader time to start
sleep 2

# Check if prompt loader started successfully
if ! kill -0 $PROMPT_PID 2>/dev/null; then
    echo "WARNING: Prompt loader failed to start, continuing without it"
else
    echo "Prompt Registry started (PID: $PROMPT_PID)"
fi

echo "Starting vLLM server..."

# Build vLLM args - customize quantization based on your model
QUANTIZATION_ARGS=""
if [ -n "${QUANTIZATION_METHOD}" ]; then
    QUANTIZATION_ARGS="--quantization ${QUANTIZATION_METHOD}"
fi

# Start vLLM (this will be the main process)
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --served-model-name "${MODEL_NAME:-brain-llm}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT:-8000}" \
    ${QUANTIZATION_ARGS} \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --trust-remote-code
