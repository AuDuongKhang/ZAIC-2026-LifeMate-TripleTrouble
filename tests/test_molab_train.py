"""Runs the training notebook's cells with fakes for the UI and subprocess, so the commands they build are checked
without marimo or a GPU."""
import ast
import json
import re
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


def _commands(tmp_path, hints, thoughts="", synth="", leave_out=0):
    """Press Start training; returns (records commands, train command)."""
    fn, params = _cell("start", "date_hints")
    (tmp_path / "data" / "train").mkdir(parents=True)
    subprocess = MagicMock()
    subprocess.run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
    fakes = {
        "Path": Path, "asr_cache": _value(""), "asr_model": _value("m"), "data_root": tmp_path / "data",
        "date_hints": _value(hints), "eval_records": _value("dev"), "examples": _value(4), "extra": _value(""),
        "hf_repo": _value("r"), "hf_token": _value(""), "hf_write": _value(""), "leave_out": _value(leave_out),
        "masked": _value(2), "max_steps": _value(0),
        "mo": MagicMock(), "perturb": _value(True), "procs": {}, "records": _value("pool"), "repo_dir": tmp_path,
        "repo_name": lambda: "", "resume": _value(False), "run_dir": lambda name: tmp_path / "ckpt" / name,
        "run_name": _value("sft-v3d"), "shlex": shlex, "start": _value(True), "step_env": dict,
        "stop": _value(False), "subprocess": subprocess, "synth": _value(synth), "sys": sys, "thoughts": _value(thoughts),
        "work": tmp_path,
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


def test_leave_app_out_names_and_flags_the_train_records_only(tmp_path):
    records, train = _commands(tmp_path, False, leave_out=50)
    assert [_after(c, "--name") for c in records] == ["gold-pool-m2-p-lao50", "gold-dev"]
    assert _after(records[0], "--leave-app-out") == "0.5" and "--leave-app-out" not in records[1]
    assert _after(train, "--records") == "gold-pool-m2-p-lao50" and _after(train, "--eval-records") == "gold-dev"


def test_leave_app_out_goes_before_the_date_hint_suffix(tmp_path):
    records, train = _commands(tmp_path, True, leave_out=25)
    assert [_after(c, "--name") for c in records] == ["gold-pool-m2-p-lao25-d", "gold-dev-d"]
    assert _after(records[0], "--leave-app-out") == "0.25"


def test_leave_app_out_off_keeps_the_commands(tmp_path):
    records, _ = _commands(tmp_path, False)
    assert [_after(c, "--name") for c in records] == ["gold-pool-m2-p", "gold-dev"]
    assert not any("--leave-app-out" in c for c in records)


def test_leave_app_out_is_off_by_default():
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    calls = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "leave_out"]
    assert len(calls) == 1
    assert {k.arg: k.value.value for k in calls[0].keywords if k.arg == "value"} == {"value": 0}


def _merge_suggestion(tmp_path, ticked, saved_records=None, thoughts=""):
    """Press Merge; returns the markdown with the suggested preset. saved_records: the run_config.json the run left."""
    fn, params = _cell("merge", "date_hints")
    mo, subprocess = MagicMock(), MagicMock()
    subprocess.run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
    if saved_records:
        (tmp_path / "sft-v3d" / "last").mkdir(parents=True)
        (tmp_path / "sft-v3d" / "last" / "run_config.json").write_text(json.dumps({"records": saved_records}), encoding="utf-8")
    fakes = {"date_hints": _value(ticked), "examples": _value(4), "json": json, "merge": _value(True), "mo": mo,
             "procs": {}, "re": re, "repo_dir": tmp_path, "repo_name": lambda: "o/lifemate-sft-v3d",
             "run_dir": lambda name: tmp_path / name, "run_name": _value("sft-v3d"), "step_env": dict,
             "subprocess": subprocess, "sys": sys, "thoughts": _value(thoughts), "work": tmp_path}
    fn(**{p: fakes[p] for p in params})
    return mo.md.call_args.args[0]


@pytest.mark.parametrize("ticked, saved, expected", [
    (True, None, True),  # no run_config.json: only the checkbox is known
    (False, None, False),
    (False, "gold-pool-m2-p-d", True),  # a new session: the box is unticked, but the run trained with hints
    (False, "gold-pool-m2-p-d-k2", True),
    (True, "gold-pool-m2-p", False),  # the run trained without hints, whatever the box says now
    (False, "gold-pool-synth-d-m2-p", False),
    (False, "gold-pool-m2-p-lao50-d", True),  # the leave-app-out suffix does not hide the date hints
    (False, "gold-pool-m2-p-lao50", False),
])
def test_merge_suggests_a_preset_that_serves_as_it_was_trained(tmp_path, ticked, saved, expected):
    assert ('"date_hints": true' in _merge_suggestion(tmp_path, ticked, saved)) is expected


def test_the_checkbox_is_off_by_default():
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    calls = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "date_hints"]
    assert len(calls) == 1
    assert {k.arg: k.value.value for k in calls[0].keywords if k.arg == "value"} == {"value": False}


@pytest.fixture
def fake_thoughts_download(tmp_path, monkeypatch):
    import huggingface_hub

    asked = []

    def fake(**kw):
        asked.append(kw)
        return str(tmp_path / "th.jsonl")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return asked


def test_thoughts_render_both_record_files_with_the_flags(tmp_path, fake_thoughts_download):
    records, train = _commands(tmp_path, False, thoughts="thoughts/r1.jsonl")
    assert all(_after(c, "--prompt") == "v3" and _after(c, "--thoughts") == str(tmp_path / "th.jsonl") for c in records)
    assert [_after(c, "--name") for c in records] == ["gold-pool-m2-p-thoughts-r1", "gold-dev-thoughts-r1"]
    assert _after(train, "--records") == "gold-pool-m2-p-thoughts-r1"
    assert _after(train, "--eval-records") == "gold-dev-thoughts-r1"
    assert [(k["repo_id"], k["repo_type"], k["filename"]) for k in fake_thoughts_download] == [("r", "dataset", "thoughts/r1.jsonl")]


def test_thoughts_keep_the_date_hints_suffix_last(tmp_path, fake_thoughts_download):
    records, _ = _commands(tmp_path, True, thoughts="thoughts/r1.jsonl")
    assert _after(records[0], "--name") == "gold-pool-m2-p-thoughts-r1-d"


def test_without_thoughts_the_records_use_no_prompt_flag(tmp_path):
    records, _ = _commands(tmp_path, False)
    assert not any("--prompt" in c or "--thoughts" in c for c in records)


def test_thoughts_and_synthetic_samples_cannot_be_combined(tmp_path, fake_thoughts_download):
    with pytest.raises(RuntimeError, match="synthetic"):
        _commands(tmp_path, False, thoughts="thoughts/r1.jsonl", synth="synth/s.jsonl")


@pytest.mark.parametrize("field, saved, expected", [
    ("thoughts/r1.jsonl", None, True),
    ("", None, False),
    ("", "gold-pool-m1-thoughts-r1", True),
    ("thoughts/r1.jsonl", "gold-pool-m1", False),
])
def test_merge_suggests_prompt_v3_for_a_run_trained_with_sentences(tmp_path, field, saved, expected):
    assert ('"prompt_version": "v3"' in _merge_suggestion(tmp_path, False, saved, thoughts=field)) is expected
