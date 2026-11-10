#!/usr/bin/env bash
# Build the solution image with the HF token as a BuildKit secret, save it as zac2026_TripleTrouble.tar.gz and print
# its MD5 (the checksum the portal asks for).
#   HF_TOKEN=... bash scripts/docker_build.sh [--build-arg BASE=pytorch/pytorch:<tag>]
#   OUT_DIR=/some/dir HF_TOKEN=... bash scripts/docker_build.sh     # where the tar.gz goes (default: repo root)
#   SAVE=0 HF_TOKEN=... bash scripts/docker_build.sh                # build only (CI streams the archive to Drive)
# Needs data/train/train.jsonl and data/train/train_tools_catalog.json (few-shot bank) in the build context, and the
# git history of configs/shipped.json's records_commit: the code that rendered the shipped model's SFT records is
# exported to docker/training_code/code_at_<commit>/ so that the image's training_code/ holds it (guide 6.2).
set -euo pipefail
cd "$(dirname "$0")/.."
py="$(command -v python3 || command -v python)"
commit="$("$py" -c 'import json; print(json.load(open("configs/shipped.json"))["records_commit"])')"
snap="docker/training_code/code_at_${commit:0:7}"
rm -rf docker/training_code/code_at_*
mkdir -p "$snap"
git archive "$commit" -- src configs notebooks scripts pyproject.toml requirements.txt requirements-train.txt \
  | tar -x -C "$snap"
echo "training code at $commit -> $snap"

tag="zac2026-tripletrouble:$(git rev-parse --short HEAD)"
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t "$tag" "$@" .
docker image ls "$tag"
if [ "${SAVE:-1}" = "0" ]; then
  echo "built $tag (not saved)"
  exit 0
fi
out="${OUT_DIR:-.}/zac2026_TripleTrouble.tar.gz"
gz=gzip
command -v pigz >/dev/null && gz=pigz
docker save "$tag" | "$gz" > "$out"
(cd "$(dirname "$out")" && md5sum "$(basename "$out")" | tee "$(basename "$out").md5")
echo "built $tag -> $out ($(du -h "$out" | cut -f1))"
