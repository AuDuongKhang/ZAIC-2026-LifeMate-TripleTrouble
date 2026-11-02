#!/usr/bin/env bash
# Build the solution image with the HF token as a BuildKit secret, then save it and print its sha256.
#   HF_TOKEN=... bash scripts/docker_build.sh [--build-arg BASE=pytorch/pytorch:<tag>]
# Needs data/train/train.jsonl and data/train/train_tools_catalog.json (few-shot bank) in the build context.
set -euo pipefail
cd "$(dirname "$0")/.."
tag="lifemate:$(git rev-parse --short HEAD)"
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t "$tag" "$@" .
out="lifemate_$(git rev-parse --short HEAD).tar.gz"
docker save "$tag" | gzip > "$out"
sha256sum "$out" | tee "$out.sha256"
echo "built $tag -> $out"
