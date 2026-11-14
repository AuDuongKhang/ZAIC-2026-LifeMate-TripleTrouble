"""Runs the preference notebook's start cell with fakes for the UI and subprocess: it must run exactly the commands of
lifemate.sft.pref_commands, chained, without marimo or a GPU."""
import ast
import os
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lifemate.sft.pref_commands import commands

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "molab_pref.py"


def _cell(*params):
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [n for n in tree.body if isinstance(n, ast.FunctionDef) and set(params) <= {a.arg for a in n.args.args}]
    assert len(cells) == 1, f"{len(cells)} cells take {params}"
    cells[0].decorator_list = []
    namespace = {}
    exec(compile(ast.Module([cells[0]], []), str(NOTEBOOK), "exec"), namespace)
    return namespace[cells[0].name], [a.arg for a in cells[0].args.args]


def v(value):
    return SimpleNamespace(value=value)


FIELDS = {"name": "pilot", "model": "TripleTrouble/lifemate-sft-q1c", "revision": "8cb4336e529f0a4b2d6f10cdcfc63d15c45b651d",
          "k": 8, "temperature": "0.7,1.0", "top_p": 0.95, "limit": 100, "lr": "1e-5", "beta": "0.1",
          "sft_weight": "1.0", "owner": "TripleTrouble", "run": "", "epochs": 2}


def _start(tmp_path, step, **override):
    fn, params = _cell("start", "step")
    (tmp_path / "data" / "train").mkdir(parents=True, exist_ok=True)
    subprocess = MagicMock()
    fakes = {"data_root": tmp_path / "data", "hf_repo": v("TripleTrouble/zaic2026-lifemate-data"), "hf_write": v("secret"),
             "mo": MagicMock(), "os": os, "procs": {}, "repo_dir": tmp_path, "shlex": shlex, "signal": signal,
             "start": v(True), "step": v(step), "step_env": dict, "stop": v(False), "subprocess": subprocess, "sys": sys,
             "time": __import__("time"),
             "work": tmp_path, **{key: v(value) for key, value in {**FIELDS, **override}.items()}}
    fn(**{p: fakes[p] for p in params})
    return subprocess.Popen.call_args.args[0]


@pytest.mark.parametrize("step", ["records", "sample", "pairs", "interpolate"])
def test_start_runs_exactly_the_commands_of_the_step(tmp_path, step):
    cmd = _start(tmp_path, step)
    f = dict(FIELDS, data_repo="TripleTrouble/zaic2026-lifemate-data")
    assert cmd == ["bash", "-c", " && ".join(shlex.join(c) for c in commands(step, f))]


def test_the_notebook_never_offers_a_test_split():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "public_test" not in text and "private_test" not in text


def test_the_monitor_draws_a_progress_bar_from_the_log(tmp_path):
    import time

    fn, params = _cell("refresh", "procs", "repo_dir")
    log = tmp_path / "pref-sample.log"
    log.write_text("old run\n", encoding="utf-8")
    start = log.stat().st_size
    with open(log, "a", encoding="utf-8") as f:
        f.write("520 points to sample (0 already done)\nprogress 130/520 points (25%), 2.0 s/point, eta 13 min\n")
    mo = MagicMock()
    process = MagicMock()
    process.poll.return_value = None
    fakes = {"hf_write": v(""), "mo": mo, "procs": {"step": process, "log": log, "log_start": start, "name": "sample",
                                                     "started": time.time()},
             "refresh": None, "repo_dir": Path(__file__).parent.parent, "sys": sys, "time": time}
    fn(**{p: fakes[p] for p in params})
    html = mo.Html.call_args.args[0]
    assert '<progress value="130" max="520"' in html and "25%" in html
    assert "old run" not in mo.ui.code_editor.call_args.kwargs["value"]


def test_the_monitor_shows_the_training_plots(tmp_path):
    import json
    import time

    pytest.importorskip("matplotlib")
    fn, params = _cell("refresh", "procs", "repo_dir")
    out = tmp_path / "checkpoints" / "sft-q1d-r1-a"
    out.mkdir(parents=True)
    rows = [{"step": s, "loss": 1.0, "lr": 1e-5, "grad_norm": 0.5, "pairs": 4, "dpo": 0.6, "nll": 0.2, "margin": 0.1,
             "pref_acc": 0.5} for s in (1, 2)]
    (out / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    log = tmp_path / "pref-train.log"
    log.write_text("step 2/40 loss 1.0\n", encoding="utf-8")
    mo = MagicMock()
    fakes = {"hf_write": v(""), "mo": mo, "procs": {"step": MagicMock(), "log": log, "log_start": 0, "name": "train",
                                                     "started": time.time(), "train_out": out},
             "refresh": None, "repo_dir": Path(__file__).parent.parent, "sys": sys, "time": time}
    fn(**{p: fakes[p] for p in params})
    assert mo.image.call_count == 6  # loss, lr, grad_norm, dpo, margin, pref_acc
    assert (out / "live_plots" / "pref_acc.png").exists()


def test_start_passes_the_epochs_field_to_train(tmp_path):
    cmd = _start(tmp_path, "train", temperature="1.0", run="d")
    f = dict(FIELDS, data_repo="TripleTrouble/zaic2026-lifemate-data", temperature="1.0", run="d")
    assert cmd == ["bash", "-c", " && ".join(shlex.join(c) for c in commands("train", f))]
    assert "--epochs 2" in cmd[2]
