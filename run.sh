#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
IMG="${DOCKERIMAGE:-deepseek-ocr}"
TAG="${DOCKERTAG:-dgx}"
NAME="${CONTAINER_NAME:-deepseek-ocr}"

# Service port (host -> container $PORT)
PORT="${PORT:-8100}"

# GPU selection: e.g. export GPU_OPT='--gpus "device=1"'
GPU_OPT="${GPU_OPT:---gpus all}"

# DEFAULT: use your provided absolute model repo path
# Override with: export MODEL_REPO="/abs/path/to/DeepSeek-OCR"
MODEL_REPO="${MODEL_REPO:-/home/rhmaomao/myServer/llm_models/DeepSeek-OCR}"

# Optional outputs dir (host) for saved artifacts
OUTPUTS_DIR="${OUTPUTS_DIR:-/home/rhmaomao/myServer/llm_services/dpocr/outputs}"

# Optional HF token (only needed for gated repos)
HF_TOKEN="${HF_TOKEN:-}"

# Optional toggles
USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# Caches & logs
ROOT_DIR="$(pwd)"
CACHE_DIR="$ROOT_DIR/cache"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/${NAME}.log"
PID_FILE="$LOG_DIR/${NAME}.logs.pid"

mkdir -p "$CACHE_DIR" "$LOG_DIR"

# ---------- Preflight ----------
if [[ ! -d "$MODEL_REPO" ]]; then
  echo "ERROR: MODEL_REPO not found: $MODEL_REPO"
  echo "Set MODEL_REPO to the DeepSeek-OCR repo directory (with config.json, *.safetensors, etc.)"
  exit 1
fi

# ---------- Build if missing ----------
if ! docker image inspect "${IMG}:${TAG}" >/dev/null 2>&1; then
  echo "Image ${IMG}:${TAG} not found; building..."
  docker build -t "${IMG}:${TAG}" .
fi

# ---------- Stop/remove old ----------
docker rm -f "$NAME" >/dev/null 2>&1 || true

# ---------- Compose docker run args ----------
RUN_ARGS=(
  -d --name "$NAME"
  ${GPU_OPT}
  --restart unless-stopped
  --log-opt max-size=50m --log-opt max-file=5
  -p "${PORT}:${PORT}"
  -v "$CACHE_DIR:/cache"
  -v "${MODEL_REPO}:/models/DeepSeek-OCR:ro"   # mount your repo
  -e HF_HOME=/cache/hf
  -e TRANSFORMERS_CACHE=/cache/hf
  -e TORCH_HOME=/cache/torch
  -e PORT="${PORT}"                            # container listens on this port
  -e REPO_ID="/models/DeepSeek-OCR"
  -e USE_FLASH_ATTENTION=0            # tell app to use the local repo
)

# Optional outputs dir
if [[ -n "$OUTPUTS_DIR" ]]; then
  mkdir -p "$OUTPUTS_DIR"
  RUN_ARGS+=(-v "${OUTPUTS_DIR}:/outputs" -e OUTPUT_DIR="/outputs")
fi

# Optional HF token
if [[ -n "$HF_TOKEN" ]]; then
  RUN_ARGS+=(-e HF_TOKEN="${HF_TOKEN}")
fi

# Optional FlashAttention toggle
if [[ -n "$USE_FLASH_ATTENTION" ]]; then
  RUN_ARGS+=(-e USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION}")
fi

# Optional CUDA_VISIBLE_DEVICES hint
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  RUN_ARGS+=(-e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
fi

# ---------- Run ----------
docker run "${RUN_ARGS[@]}" "${IMG}:${TAG}"

# ---------- Start nohup log follower ----------
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  kill "$(cat "$PID_FILE")" || true
fi
nohup bash -c "docker logs -f --since=0s ${NAME}" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

# ---------- Info ----------
echo "✅ ${NAME} started."
echo "   API:        http://localhost:${PORT}/docs"
echo "   Health:     http://localhost:${PORT}/healthz"
echo "   Logs:       $LOG_FILE"
echo "   Log PID:    $(cat "$PID_FILE")"
echo "   Model:      ${MODEL_REPO}  ->  /models/DeepSeek-OCR (REPO_ID)"
if [[ -n "$OUTPUTS_DIR" ]]; then
  echo "   Outputs:    $OUTPUTS_DIR <- /outputs"
fi
echo
echo "Follow live:   tail -f \"$LOG_FILE\""
echo "Stop logs:     kill \$(cat \"$PID_FILE\")"
echo "Stop service:  docker rm -f ${NAME}"

