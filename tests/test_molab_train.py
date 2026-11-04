"""Runs the training notebook's cells with fakes for the UI and subprocess, so the commands they build are checked
without marimo or a GPU."""
import ast
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "molab_train.py"


def _cell(*params):
    """The notebook cell function that takes all of `params`."""
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [n for n in tree.body if isinstance(n, ast.FunctionDef) and set(params) <= {a.arg for a in n.args.args}]
    assert len(cells) == 1, f"{len(cells)} cells take {params}"
    cells[0].decorator_list = []
    namespace = {}
    exec(compile(ast.Module([cells[0]], []), str(NOTEBOOK), "exec"), namespace)
    return namespace[cells[0].name], [a.arg for a in cells[0].args.args]


def _value(value):
    return SimpleNamespace(value=value)


def _commands(tmp_path, hints):
    """Press Start training; returns (records commands, train command)."""
    fn, params = _cell("start", "date_hints")
    (tmp_path / "data" / "train").mkdir(parents=True)
    subprocess = MagicMock()
    subprocess.run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
    fakes = {
        "Path": Path, "asr_cache": _value(""), "asr_model": _value("m"), "data_root": tmp_path / "data",
        "date_hints": _value(hints), "eval_records": _value("dev"), "examples": _value(4), "extra": _value(""),
        "hf_repo": _value("r"), "hf_token": _value(""), "hf_write": _value(""), "masked": _value(2), "max_steps": _value(0),
        "mo": MagicMock(), "perturb": _value(True), "procs": {}, "records": _value("pool"), "repo_dir": tmp_path,
        "repo_name": lambda: "", "resume": _value(False), "run_dir": lambda name: tmp_path / "ckpt" / name,
        "run_name": _value("sft-v3d"), "shlex": shlex, "start": _value(True), "step_env": dict,
        "stop": _value(False), "subprocess": subprocess, "synth": _value(""), "sys": sys, "work": tmp_path,
    }
    fn(**{p: fakes[p] for p in params})
    return [c.args[0] for c in subprocess.run.call_args_list], subprocess.Popen.call_args.args[0]


def _after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_ticking_date_hints_renders_both_record_files_with_the_flag(tmp_path):
    records, train = _commands(tmp_path, True)
    assert len(records) == 2 and all("--date-hints" in c for c in records)
    assert [_after(c, "--name") for c in records] == ["gold-pool-m2-p-d", "gold-dev-d"]
    assert _after(train, "--records") == "gold-pool-m2-p-d" and _after(train, "--eval-records") == "gold-dev-d"


def test_without_date_hints_the_commands_stay_as_they_were(tmp_path):
    records, train = _commands(tmp_path, False)
    assert not any("--date-hints" in c for c in records)
    assert [_after(c, "--name") for c in records] == ["gold-pool-m2-p", "gold-dev"]
    assert _after(train, "--records") == "gold-pool-m2-p" and _after(train, "--eval-records") == "gold-dev"


@pytest.mark.parametrize("hints, expected", [(True, True), (False, False)])
def test_merge_suggests_a_preset_that_serves_as_it_was_trained(tmp_path, hints, expected):
    fn, params = _cell("merge", "date_hints")
    mo, subprocess = MagicMock(), MagicMock()
    subprocess.run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
    fakes = {"date_hints": _value(hints), "examples": _value(4), "merge": _value(True), "mo": mo, "procs": {},
             "repo_dir": tmp_path, "repo_name": lambda: "o/lifemate-sft-v3d", "run_dir": lambda name: tmp_path / name,
             "run_name": _value("sft-v3d"), "step_env": dict, "subprocess": subprocess, "sys": sys,
             "work": tmp_path}
    fn(**{p: fakes[p] for p in params})
    assert ('"date_hints": true' in mo.md.call_args.args[0]) is expected
