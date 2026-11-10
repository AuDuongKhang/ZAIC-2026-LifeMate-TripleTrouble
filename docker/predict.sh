#!/usr/bin/env bash
# Organiser entry: read the split folder under /data, write /result/submission.jsonl (one line per sample, with
# `time` in ms), print the model load and predict times. Offline; the models are baked into the image.
#   docker run --gpus '"device=0"' -v <data>:/data:ro -v <result>:/result <image> /bin/bash /code/predict.sh
set -euo pipefail
cd /code
python -m lifemate.docker_entry --data "${LIFEMATE_DATA:-/data}" --out "${LIFEMATE_RESULT:-/result}/submission.jsonl"
