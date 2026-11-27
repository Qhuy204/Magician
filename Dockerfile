# syntax=docker/dockerfile:1
# Build:
#   NVIDIA: docker build -t magician:cuda .
#   AMD:    docker build -t magician:rocm --build-arg TORCH_VARIANT=therock .
#           (fallback cũ: TORCH_VARIANT=rocm6.4 XTTS_TORCH_VARIANT=rocm6.2 — torch 2.8 wheel pytorch.org,
#            gfx1032/RX6600 phải ép HSA_OVERRIDE_GFX_VERSION=10.3.0)
#   CPU:    docker build -t magician:cpu  --build-arg TORCH_VARIANT=cpu --build-arg XTTS_TORCH_VARIANT=cpu .
# Thêm engine XTTS (env riêng, +~5GB): --build-arg INSTALL_XTTS=1
# Apple Silicon: Docker trên Mac không truy cập được GPU (Metal), chạy native bằng scripts/run_mac.sh.
# LƯU Ý: image này chỉ dùng được GPU AMD khi chạy trên Linux host có /dev/kfd|dri —
# Docker Desktop trên Windows không passthrough được GPU AMD vào container, bất kể torch bên trong là gì.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG TORCH_VARIANT=cu128
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=
ARG INSTALL_XTTS=0
ARG XTTS_TORCH_VARIANT=cu121
# TheRock (ROCm chính thức của AMD, torch 2.14+): kernel native cho cả gfx1032/RX6600.
# THEROCK_DEVICE=device-all an toàn mọi card; thu hẹp theo GPU (vd device-gfx1032) cho image nhẹ hơn.
ARG THEROCK_INDEX=https://stable.repo.amd.com/rocm/pytorch/whl-next/
# rocm meta-package + rocm-sdk-device-gfx* (runtime ROCm) nằm ở index core/, không phải pytorch/
ARG THEROCK_DEVICE=device-all
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt requirements-xtts.txt /tmp/
# Cài torch bản GPU trước rồi khoá version: easyocr kéo torchvision từ PyPI sẽ ghi đè torch nếu không pin
RUN python -m venv /opt/venv \
    && if [ "$TORCH_VARIANT" = "therock" ]; then \
         /opt/venv/bin/pip install --index-url "$THEROCK_INDEX" \
           --extra-index-url https://stable.repo.amd.com/rocm/core/whl-next/ \
           --extra-index-url https://pypi.org/simple \
           "torch[${THEROCK_DEVICE}]==${TORCH_VERSION}" \
           "torchvision[${THEROCK_DEVICE}]==${TORCHVISION_VERSION}" \
           "torchaudio==${TORCHAUDIO_VERSION:-$TORCH_VERSION}"; \
       else \
         /opt/venv/bin/pip install torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION:-$TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
           --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}; \
       fi \
    && /opt/venv/bin/pip freeze | grep -E "^(torch|torchaudio|torchvision)==" > /tmp/torch-pin.txt \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt -c /tmp/torch-pin.txt
RUN mkdir -p /opt/xtts && if [ "$INSTALL_XTTS" = "1" ]; then \
      python -m venv /opt/xtts \
      && /opt/xtts/bin/pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/${XTTS_TORCH_VARIANT} \
      && /opt/xtts/bin/pip install -r /tmp/requirements-xtts.txt; \
    fi

FROM python:${PYTHON_VERSION}-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 magician \
    && mkdir -p /models/app /app/data && chown -R magician /models /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/xtts /opt/xtts
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    MODELS_DIR=/models/app \
    TTS_HOME=/models/tts \
    TORCH_HOME=/models/torch \
    HF_HOME=/models/hf \
    DEVICE=auto \
    TTS_ENGINE=omnivoice \
    XTTS_PYTHON=/opt/xtts/bin/python
WORKDIR /app
COPY --chown=magician app ./app
# Container start as root để entrypoint chỉnh group theo GID thật của /dev/kfd|dri (ROCm),
# rồi drop xuống magician — xem docker-entrypoint.sh
USER magician

# Tải sẵn model (Demucs, OmniVoice, EasyOCR ar/hi/pt, LaMa) vào image: --build-arg PRELOAD_MODELS=1
ARG PRELOAD_MODELS=0
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python -c "from demucs.pretrained import get_model; get_model('htdemucs'); from huggingface_hub import snapshot_download; snapshot_download('k2-fsa/OmniVoice'); import easyocr; [easyocr.Reader(l, gpu=False, model_storage_directory='/models/app/easyocr') for l in (['ar', 'en'], ['hi', 'en'], ['pt', 'en'])]" \
      && curl -fsSL -o /models/app/big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt; \
    fi

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
USER root
VOLUME ["/models", "/app/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:8000/health || exit 1
ENTRYPOINT ["bash", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
