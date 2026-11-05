"""Runs the sentence-writer notebook's start cell with fakes for the UI and subprocess, so the command it builds is
checked without marimo or a GPU."""
import ast
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "molab_thoughts.py"


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


def _start(tmp_path, name, subset, revision, limit, teacher="Qwen/Qwen3.5-27B"):
    """Press Start; returns the writer command."""
    fn, params = _cell("start", "subset")
    (tmp_path / "data" / "train").mkdir(parents=True)
    subprocess = MagicMock()
    fakes = {
        "Path": Path, "data_root": tmp_path / "data", "hf_repo": _value("o/data"), "hf_write": _value("secret"),
        "limit": _value(limit), "mo": MagicMock(), "name": _value(name), "procs": {}, "repo_dir": tmp_path,
        "revision": _value(revision), "shlex": shlex, "start": _value(True), "step_env": dict, "stop": _value(False),
        "subprocess": subprocess, "subset": _value(subset), "sys": sys, "teacher": _value(teacher),
        "thoughts_dir": tmp_path / "thoughts", "work": tmp_path,
    }
    fn(**{p: fakes[p] for p in params})
    return subprocess.Popen.call_args.args[0]


def _after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_start_builds_the_writer_command(tmp_path):
    cmd = _start(tmp_path, name="pilot1", subset="pool", revision="abc", limit=100)
    assert cmd[1:3] == ["-m", "lifemate.sft.thoughts"]
    assert _after(cmd, "--name") == "pilot1" and _after(cmd, "--subset") == "pool" and _after(cmd, "--revision") == "abc"
    assert _after(cmd, "--limit") == "100" and _after(cmd, "--teacher") == "Qwen/Qwen3.5-27B" and "--hf-repo" in cmd


def test_no_limit_means_the_whole_subset(tmp_path):
    assert "--limit" not in _start(tmp_path, name="r1", subset="dev", revision="abc", limit=0)


def test_a_blank_revision_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="revision"):
        _start(tmp_path, name="r1", subset="pool", revision=" ", limit=0)


def test_the_notebook_never_offers_a_test_split():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "public_test" not in text and "private_test" not in text


def test_the_subset_choices_are_pool_dev_holdout_and_pool_is_the_default():
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    dropdowns = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "dropdown"]
    assert len(dropdowns) == 1
    options = ast.literal_eval(dropdowns[0].args[0])
    assert sorted(options.values()) == ["dev", "holdout", "pool"]
    assert options[ast.literal_eval(next(k.value for k in dropdowns[0].keywords if k.arg == "value"))] == "pool"
