#!/usr/bin/env bash
# Build the solution image with the HF token as a BuildKit secret, save it as zac2026_TripleTrouble.tar.gz and print
# its MD5 (the checksum the portal asks for).
#   HF_TOKEN=... bash scripts/docker_build.sh [--build-arg BASE=pytorch/pytorch:<tag>]
#   OUT_DIR=/mnt/out HF_TOKEN=... bash scripts/docker_build.sh      # where the tar.gz goes (default: repo root)
# Needs data/train/train.jsonl and data/train/train_tools_catalog.json (few-shot bank) in the build context.
set -euo pipefail
cd "$(dirname "$0")/.."
tag="zac2026-tripletrouble:$(git rev-parse --short HEAD)"
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t "$tag" "$@" .
out="${OUT_DIR:-.}/zac2026_TripleTrouble.tar.gz"
gz=gzip
command -v pigz >/dev/null && gz=pigz
docker save "$tag" | "$gz" > "$out"
(cd "$(dirname "$out")" && md5sum "$(basename "$out")" | tee "$(basename "$out").md5")
docker image ls "$tag"
echo "built $tag -> $out ($(du -h "$out" | cut -f1))"
