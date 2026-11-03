# syntax=docker/dockerfile:1
# LifeMate solution image: offline inference with the presets and model revisions in configs/shipped.json.
#   scripts/docker_build.sh                       # build with the HF token as a build secret, save, checksum
#   docker run --rm --gpus all --network none -v <split folder>:/data:ro -v <out>:/app/result lifemate:<tag>
# BASE must ship torch + torchvision built together (requirements.txt adds no torch); match molab's torch version and
# stay on CUDA 12.x so the image runs on the scoring A30's driver.
ARG BASE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE}

ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps -e .

# Model weights at the pinned revisions; the token is a build secret, never stored in a layer.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null)" \
    python -c "from lifemate.docker_entry import fetch_models, load_config; fetch_models(load_config())"

# Few-shot bank: train requests and gold calls (not audio, not environments). Training code and docs ship too.
COPY data/train/train.jsonl data/train/train_tools_catalog.json ./data/train/
COPY notebooks ./notebooks
COPY scripts ./scripts
COPY tests ./tests
COPY README.md RESULTS.md CLAUDE.md requirements-train.txt requirements-dev.txt ./

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "-m", "lifemate.docker_entry"]
