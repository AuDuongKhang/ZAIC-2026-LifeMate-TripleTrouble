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
          "sft_weight": "1.0", "owner": "TripleTrouble", "run": ""}


def _start(tmp_path, step):
    fn, params = _cell("start", "step")
    (tmp_path / "data" / "train").mkdir(parents=True, exist_ok=True)
    subprocess = MagicMock()
    fakes = {"data_root": tmp_path / "data", "hf_repo": v("TripleTrouble/zaic2026-lifemate-data"), "hf_write": v("secret"),
             "mo": MagicMock(), "os": os, "procs": {}, "repo_dir": tmp_path, "shlex": shlex, "signal": signal,
             "start": v(True), "step": v(step), "step_env": dict, "stop": v(False), "subprocess": subprocess, "sys": sys,
             "work": tmp_path, **{key: v(value) for key, value in FIELDS.items()}}
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
