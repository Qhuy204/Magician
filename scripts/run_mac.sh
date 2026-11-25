#!/usr/bin/env bash
# Chạy native trên Apple Silicon (Docker trên Mac không dùng được GPU Metal/MPS).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || brew install ffmpeg
if [ ! -d .venv ]; then
  python3.11 -m venv .venv
  .venv/bin/pip install torch==2.8.0 torchaudio==2.8.0
  .venv/bin/pip install -r requirements.txt
fi
export DEVICE="${DEVICE:-auto}" PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
