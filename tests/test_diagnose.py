import json

import pytest

from lifemate.data import default_data_root, load_catalog, load_samples
from lifemate.diagnose import diagnose
from lifemate.evaluate import label_kind

ROOT = default_data_root()
pytestmark = pytest.mark.skipif(not (ROOT / "train").exists(), reason="needs data/train")


def _write(run, preds, traces=None):
    run.mkdir()
    (run / "submission.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")
    if traces:
        (run / "trace.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces), encoding="utf-8")


def _with_extra_optional(samples, catalog):
    """(sample, call index, optional param the gold call omits) for the first act sample that has one."""
    for s in samples:
        if label_kind(s["label"]["tool_calls"]) != "act":
            continue
        for i, call in enumerate(s["label"]["tool_calls"]):
            props = catalog[call["name"]]["parameters"].get("properties", {})
            required = set(catalog[call["name"]]["parameters"].get("required", []))
            spare = [p for p in props if p not in required and p not in call["arguments"]]
            if spare:
                return s, i, spare[0]
    raise AssertionError("no sample with an omitted optional param")


def test_diagnose_counts_a_call_with_an_extra_optional_key_as_matched(tmp_path):
    # organiser FAQ: only the gold call's arguments are compared, so an extra optional argument still matches
    samples, catalog = load_samples(ROOT, "train"), load_catalog(ROOT, "train")
    exact = next(s for s in samples if label_kind(s["label"]["tool_calls"]) == "act" and s["id"])
    extra, i, param = _with_extra_optional([s for s in samples if s["id"] != exact["id"]], catalog)
    calls = [dict(c, arguments=dict(c["arguments"])) for c in extra["label"]["tool_calls"]]
    calls[i]["arguments"][param] = "x"
    _write(tmp_path / "run", [{"id": exact["id"], "tool_calls": exact["label"]["tool_calls"]},
                              {"id": extra["id"], "tool_calls": calls}])
    r = diagnose(tmp_path / "run", ROOT)
    n_exact, n_extra = len(exact["label"]["tool_calls"]), len(extra["label"]["tool_calls"])
    assert r["gold_calls"] == n_exact + n_extra
    assert r["matched"] == n_exact + n_extra
    assert r["calls_with_extra_key"] == 0


def test_diagnose_missing_calls_and_call_counts(tmp_path):
    samples, catalog = load_samples(ROOT, "train"), load_catalog(ROOT, "train")
    s = next(x for x in samples if label_kind(x["label"]["tool_calls"]) == "act" and len(x["label"]["tool_calls"]) >= 2)
    first = s["label"]["tool_calls"][0]
    _write(tmp_path / "run", [{"id": s["id"], "tool_calls": [first]}],
           [{"id": s["id"], "agent": {"steps": [], "stop": "no_call"}}])
    r = diagnose(tmp_path / "run", ROOT)
    missing = r["missing_read"] + r["missing_write"] + r["missing_ask"]
    assert r["matched"] == 1 and missing == len(s["label"]["tool_calls"]) - 1
    assert r["fewer_calls"] == 1 and r["more_calls"] == 0
    assert r["stop"] == {"no_call": 1}


def test_diagnose_counts_calls_on_out_of_scope_samples_separately(tmp_path):
    samples = load_samples(ROOT, "train")
    s = next(x for x in samples if label_kind(x["label"]["tool_calls"]) == "empty")
    _write(tmp_path / "run", [{"id": s["id"], "tool_calls": [{"name": "alarm.list", "arguments": {}}]}])
    r = diagnose(tmp_path / "run", ROOT)
    assert r["out_of_scope_with_calls"] == 1 and r["extra_pred_calls"] == 0 and r["gold_calls"] == 0


def test_diagnose_counts_predicted_calls_outside_gold(tmp_path):
    samples = load_samples(ROOT, "train")
    s = next(x for x in samples if label_kind(x["label"]["tool_calls"]) == "act")
    gold = s["label"]["tool_calls"]
    stray = {"name": "no.such_tool", "arguments": {}}
    _write(tmp_path / "run", [{"id": s["id"], "tool_calls": gold + [stray]}])
    r = diagnose(tmp_path / "run", ROOT)
    assert r["matched"] == len(gold) and r["extra_pred_calls"] == 1 and r["more_calls"] == 1


def test_diagnose_refuses_runs_that_are_not_train(tmp_path):
    run = tmp_path / "run"
    _write(run, [])
    (run / "meta.json").write_text(json.dumps({"split": "public_test"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="train"):
        diagnose(run, ROOT)
