#!/usr/bin/env bash
# CPU smoke test of the solution image, run the way the organisers run it but without a GPU: --network none, the split
# mounted read-only under /data, results in /result. Uses configs/smoke.json (empty agent, no ASR) for the runs, so it
# checks the plumbing (layout, offline weights, predict.sh, notebook, output format), not the score.
#   bash scripts/docker_smoke.sh zac2026-tripletrouble:<sha> data/public_test
set -euo pipefail
image="$1"
split_dir="$(cd "$2" && pwd)"
split="$(basename "$split_dir")"
out="$(mktemp -d)"
chmod 777 "$out"
run=(docker run --rm --network none -v "$split_dir:/data/$split:ro" -v "$out:/result"
     -e LIFEMATE_SHIPPED=/code/configs/smoke.json "$image")

echo "== shipped models are in the image (offline)"
docker run --rm --network none "$image" python -m lifemate.docker_entry --check-models

echo "== predict.sh"
"${run[@]}" /bin/bash /code/predict.sh

echo "== predict_notebook.ipynb"
"${run[@]}" jupyter nbconvert --to notebook --execute /code/predict_notebook.ipynb \
    --ExecutePreprocessor.timeout=-1 --output-dir /tmp --output predict_notebook_run

echo "== compare the two result files"
python3 - "$out" "$split_dir/$split.jsonl" <<'PY'
import json, sys
from pathlib import Path

out, samples = Path(sys.argv[1]), Path(sys.argv[2])
n = sum(1 for line in samples.read_text(encoding="utf-8").splitlines() if line.strip())
def read(name):
    return [json.loads(line) for line in (out / name).read_text(encoding="utf-8").splitlines() if line.strip()]
sub, nb = read("submission.jsonl"), read("jupyter_submission.jsonl")
assert len(sub) == len(nb) == n, (len(sub), len(nb), n)
for a, b in zip(sub, nb):
    assert set(a) == set(b) == {"id", "tool_calls", "state_hash", "time"}, (a.keys(), b.keys())
    assert isinstance(a["time"], int) and isinstance(b["time"], int)
    assert (a["id"], a["tool_calls"], a["state_hash"]) == (b["id"], b["tool_calls"], b["state_hash"]), a["id"]
print(f"ok: {n} lines in both files, same calls and hashes")
PY

echo "== zaic_tools --validate"
(cd "$(dirname "$split_dir")" && PYTHONPATH="$split" python3 -m zaic_tools --split "$split" --root . \
    --validate "$out/submission.jsonl")
echo "smoke test passed ($out)"
