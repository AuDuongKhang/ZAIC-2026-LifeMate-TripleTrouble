import json

import pytest

from lifemate.agents.ask_guard import SYS_TOOL, candidate_lists
from lifemate.data import default_data_root, import_zaic_tools, load_catalog
from lifemate.evaluate import evaluate
from lifemate.guard_sim import simulate
from lifemate.splits import load_splits


def _picks_from_a_list(sample, catalog, zaic_tools, root):
    """True when a gold write passes an x-ref id taken from an earlier list of >= 2 candidates."""
    env = zaic_tools.ToolEnv("train", sample["id"], root=root)
    cands = {}
    for call in sample["label"]["tool_calls"]:
        spec = catalog.get(call["name"], {})
        props = spec.get("parameters", {}).get("properties", {})
        if spec.get("kind") == "write" and any(props.get(k, {}).get("x-ref") and v in cands
                                               for k, v in call["arguments"].items() if isinstance(v, str)):
            return True
        candidate_lists(env.call(call["name"], call["arguments"]), cands)
    return False


@pytest.fixture(scope="module")
def gold_run(train_samples, tmp_path_factory):
    """A fake run on 6 pool samples whose trace is their gold calls; 3 of them pick an id from a list."""
    root = default_data_root()
    zaic_tools, catalog = import_zaic_tools(root, "train"), load_catalog(root, "train")
    pool = set(load_splits()["pool"])
    picks, others = [], []
    for s in train_samples:
        if s["id"] not in pool or not s["label"]["tool_calls"]:
            continue
        (picks if _picks_from_a_list(s, catalog, zaic_tools, root) else others).append(s)
        if len(picks) >= 3 and len(others) >= 3:
            break
    run = tmp_path_factory.mktemp("run")
    with open(run / "trace.jsonl", "w", encoding="utf-8") as t, open(run / "submission.jsonl", "w", encoding="utf-8") as f:
        for s in picks[:3] + others[:3]:
            env = zaic_tools.ToolEnv("train", s["id"], root=root)
            calls = [dict(c, result=env.call(c["name"], c["arguments"])) for c in s["label"]["tool_calls"]]
            t.write(json.dumps({"id": s["id"], "request": s["audio_text"], "calls": calls}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"id": s["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()},
                               ensure_ascii=False) + "\n")
    evaluate(run / "submission.jsonl", "train", root)
    return run, {s["id"] for s in picks[:3]}


def test_a_guard_that_never_fires_changes_nothing(gold_run):
    run, _ = gold_run
    r = simulate(run, default_data_root(), min_score=0.0, margin=-1.0)
    assert r["fired"] == 0 and r["diff"] == 0 and r["same"] == 6


def test_a_guard_that_always_fires_ends_the_picking_samples_on_an_ask(gold_run):
    run, picking = gold_run
    r = simulate(run, default_data_root(), min_score=2.0, margin=0.0)
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(r["out"] / "submission.jsonl", encoding="utf-8")}
    assert r["fired"] == 3
    for sid in picking:
        assert rows[sid]["tool_calls"][-1]["name"] == SYS_TOOL
    assert (r["out"] / "per_sample.jsonl").exists()
