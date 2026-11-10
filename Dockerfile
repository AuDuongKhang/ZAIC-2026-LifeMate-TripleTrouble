# syntax=docker/dockerfile:1
# LifeMate solution image (organiser Docker guide layout): code in /code, input under /data, output in /result.
#   HF_TOKEN=... bash scripts/docker_build.sh       # build with the HF token as a build secret, save, MD5
#   docker run --rm --gpus '"device=0"' --network none -v <folder holding private_test/>:/data:ro -v <out>:/result \
#       zac2026-tripletrouble:<sha> /bin/bash /code/predict.sh
# BASE ships torch + torchvision built together (requirements.txt adds no torch). CUDA must stay <= 13.2 (organiser
# driver 595) and torch must carry kernels for both the A30 (sm_80) and the RTX 5090 (sm_120).
ARG BASE=pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime
FROM ${BASE}

ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Ho_Chi_Minh \
    HF_HOME=/models
WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN python -c "import torch; a = torch._C._cuda_getArchFlags().split(); print(torch.__version__, torch.version.cuda, a); assert 'sm_80' in a and 'sm_120' in a, a"

COPY requirements.txt pyproject.toml ./
COPY docker/constraints.txt ./docker/constraints.txt
RUN pip install -r requirements.txt -c docker/constraints.txt jupyterlab nbconvert ipykernel
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps -e .

# Fine-tuned weights at the pinned revisions (no base model); the token is a build secret, never stored in a layer.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null)" HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    python -c "from lifemate.docker_entry import fetch_models, load_config; fetch_models(load_config())" \
    && rm -rf /models/xet && du -sh /models/*

# Few-shot bank: train requests and gold calls (not audio, not environments). Training code and docs ship too.
COPY data/train/train.jsonl data/train/train_tools_catalog.json ./data/train/
COPY notebooks ./notebooks
COPY scripts ./scripts
COPY tests ./tests
COPY docs/shipped-model.md ./docs/shipped-model.md
COPY README.md RESULTS.md CLAUDE.md requirements-train.txt requirements-dev.txt ./
COPY docker/training_code ./training_code
COPY docker/predict.sh docker/start_jupyter.sh docker/predict_notebook.ipynb ./
RUN chmod +x predict.sh start_jupyter.sh

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 9777
CMD ["bash", "/code/predict.sh"]
