"""The timing notebook and predict.sh (docker_entry) must give the same submission on a real split."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lifemate.data import REPO_ROOT, default_data_root

NOTEBOOK = REPO_ROOT / "docker" / "predict_notebook.ipynb"
SMOKE = REPO_ROOT / "configs" / "smoke.json"


def _code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_notebook_has_the_guide_steps_and_never_writes_submission_jsonl():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4  # seed, load models, read test cases, predict
    assert all(not c["outputs"] for c in code_cells)
    code = _code(NOTEBOOK)
    assert "seed_everything" in "".join(code_cells[0]["source"])
    assert "jupyter_submission.jsonl" in code and '"submission.jsonl"' not in code


def test_notebook_matches_predict_on_public_test(tmp_path):
    split_dir = default_data_root() / "public_test"
    if not (split_dir / "public_test.jsonl").is_file():
        pytest.skip("needs data/public_test")
    mounted = tmp_path / "data"
    mounted.mkdir()
    try:
        os.symlink(split_dir.resolve(), mounted / "public_test", target_is_directory=True)
    except OSError as e:
        pytest.skip(f"symlinks not available: {e}")
    result = tmp_path / "result"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONUTF8": "1", "LIFEMATE_DATA": str(mounted),
           "LIFEMATE_RESULT": str(result), "LIFEMATE_SHIPPED": str(SMOKE)}
    subprocess.run([sys.executable, "-m", "lifemate.docker_entry", "--data", str(mounted),
                    "--out", str(result / "submission.jsonl"), "--work", str(tmp_path / "work")],
                   env=env, check=True, cwd=REPO_ROOT)
    script = tmp_path / "notebook.py"
    script.write_text(_code(NOTEBOOK), encoding="utf-8")
    subprocess.run([sys.executable, str(script)], env=env, check=True, cwd=REPO_ROOT)

    sub, nb = _read(result / "submission.jsonl"), _read(result / "jupyter_submission.jsonl")
    n = len((split_dir / "public_test.jsonl").read_text(encoding="utf-8").splitlines())
    assert len(sub) == len(nb) == n
    for a, b in zip(sub, nb):
        assert set(a) == set(b) == {"id", "tool_calls", "state_hash", "time"}
        assert isinstance(a["time"], int) and isinstance(b["time"], int)
        assert (a["id"], a["tool_calls"], a["state_hash"]) == (b["id"], b["tool_calls"], b["state_hash"])
