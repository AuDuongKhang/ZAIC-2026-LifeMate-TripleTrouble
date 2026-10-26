"""End-to-end on a few train samples: the oracle must score 100, which checks runner + scorer together.
Runs in a subprocess because each split has its own zaic_tools under the same module name."""
import json
import os
import subprocess
import sys

import pytest

from lifemate.data import REPO_ROOT, default_data_root

pytestmark = pytest.mark.skipif(not (default_data_root() / "train").is_dir(), reason="data/train not downloaded")


def test_oracle_scores_100_on_train(tmp_path):
    out = tmp_path / "run"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    subprocess.run([sys.executable, "-m", "lifemate.infer", "--split", "train", "--agent", "oracle", "--asr", "gold",
                    "--limit", "40", "--out", str(out)], check=True, env=env, cwd=REPO_ROOT)
    subprocess.run([sys.executable, "-m", "lifemate.evaluate", str(out)], check=True, env=env, cwd=REPO_ROOT)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["score"] == 100.0
