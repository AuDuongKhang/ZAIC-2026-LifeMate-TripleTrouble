import copy
import json

import pytest

from lifemate.data import default_data_root, load_catalog
from lifemate.sft import recovery
from lifemate.sft.recovery import recovery_sample
from lifemate.splits import load_splits

CATALOG = {"m.find": {"kind": "read"}, "m.list": {"kind": "read"}, "m.send": {"kind": "write"},
           "sys.ask_missing_info": {"kind": "meta"}}


def call(name, **args):
    return {"name": name, "arguments": args}


def trace(*calls):
    return {"id": "s1", "request": "x", "calls": [dict(c, result={"ok": True}) for c in calls]}


SAMPLE = {"id": "s1", "audio": "a.wav", "audio_text": "gửi tin cho Minh", "context": {"installed_apps": {}}, "history": [],
          "label": {"tool_calls": [call("m.find", q="Minh"), call("m.send", to="c1")], "tool_results": [{}, {}]}}


def test_recovery_keeps_the_leading_reads_of_a_wrong_trace():
    s = recovery_sample(SAMPLE, trace(call("m.find", q="Min"), call("m.list"), call("m.send", to="c1")), CATALOG, 0)
    assert s["id"] == "s1~r0" and s["env_id"] == "s1" and s["source"] == "recovery"
    assert s["prefix_calls"] == [call("m.find", q="Min"), call("m.list")]
    assert s["label"] == SAMPLE["label"] and s["audio_text"] == SAMPLE["audio_text"] and s["history"] == []
    assert "prefix_calls" not in SAMPLE  # the base sample is not modified


def test_recovery_stops_prefix_at_a_write():
    s = recovery_sample(SAMPLE, trace(call("m.find", q="Min"), call("m.send", to="c9"), call("m.list")), CATALOG, 1)
    assert s["prefix_calls"] == [call("m.find", q="Min")] and s["id"] == "s1~r1"


def test_recovery_stops_prefix_at_an_ask():
    ask = call("sys.ask_missing_info", tool_name="m.send", params=["to"])
    s = recovery_sample(SAMPLE, trace(call("m.list"), ask), CATALOG, 0)
    assert s["prefix_calls"] == [call("m.list")]


def test_recovery_none_when_on_path():
    assert recovery_sample(SAMPLE, trace(*SAMPLE["label"]["tool_calls"]), CATALOG, 0) is None
    assert recovery_sample(SAMPLE, trace(call("m.find", q="Minh"), call("m.send", to="zzz")), CATALOG, 0) is None  # reads match


def test_recovery_none_without_a_leading_read():
    assert recovery_sample(SAMPLE, trace(call("m.send", to="c9")), CATALOG, 0) is None
    assert recovery_sample(SAMPLE, trace(), CATALOG, 0) is None


def test_recovery_counts_a_read_after_the_gold_prefix_as_a_difference():
    gold_reads_then_extra = trace(call("m.find", q="Minh"), call("m.list"), call("m.send", to="c1"))
    s = recovery_sample(SAMPLE, gold_reads_then_extra, CATALOG, 0)
    assert s["prefix_calls"] == [call("m.find", q="Minh"), call("m.list")]  # the second read has no gold twin at its position


def test_recovery_does_not_modify_its_input():
    t = trace(call("m.find", q="Min"))
    before = copy.deepcopy((SAMPLE, t))
    recovery_sample(SAMPLE, t, CATALOG, 0)
    assert (SAMPLE, t) == before


def _real_run(tmp_path, samples):
    """A run folder whose trace holds, for each sample, its first gold read with a changed argument."""
    catalog = load_catalog(default_data_root(), "train")
    run = tmp_path / "run"
    run.mkdir()
    lines = []
    for s in samples:
        gold = s["label"]["tool_calls"]
        bad = copy.deepcopy(gold[0])
        bad["arguments"] = dict(bad["arguments"], zz_extra="changed")
        lines.append({"id": s["id"], "request": s["audio_text"], "calls": [dict(bad, result={"ok": True})] + [dict(c, result={}) for c in gold[1:]]})
    (run / "trace.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    return run, catalog


def test_cli_writes_only_verified_recovery_samples(train_samples, tmp_path):
    catalog = load_catalog(default_data_root(), "train")
    pool = set(load_splits()["pool"])
    picked = [s for s in train_samples if s["id"] in pool and s["label"]["tool_calls"]
              and (catalog.get(s["label"]["tool_calls"][0]["name"]) or {}).get("kind") == "read"][:8]
    assert picked
    run, _ = _real_run(tmp_path, picked)
    out = recovery.main(["--run", str(run), "--name", "t", "--max", "5", "--out-dir", str(tmp_path / "synth")])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert out.name == "recovery-t.jsonl" and 0 < len(rows) <= 5
    for r in rows:
        assert r["env_id"] in pool and r["prefix_calls"] and r["id"].split("~r")[0] == r["env_id"]
        assert r["prefix_calls"][0]["arguments"]["zz_extra"] == "changed"
    report = json.loads((tmp_path / "synth" / "recovery-t.report.json").read_text(encoding="utf-8"))
    assert report["usable"] == len(rows) and report["candidates"] >= len(rows)


def test_cli_refuses_a_run_that_is_not_a_pool_run(train_samples, tmp_path):
    holdout = set(load_splits()["holdout"])
    s = next(x for x in train_samples if x["id"] in holdout and x["label"]["tool_calls"])
    run, _ = _real_run(tmp_path, [s])
    with pytest.raises(SystemExit):
        recovery.main(["--run", str(run), "--name", "t", "--out-dir", str(tmp_path / "synth")])
