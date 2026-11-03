import json

import pytest

from lifemate.data import default_data_root, import_zaic_tools
from lifemate.evaluate import evaluate
from lifemate.sim import rewrite_run
from lifemate.splits import load_splits


@pytest.fixture(scope="module")
def gold_run(train_samples, tmp_path_factory):
    """A fake run on 6 pool samples with gold calls (act samples only) as its trace and submission."""
    root = default_data_root()
    zaic_tools = import_zaic_tools(root, "train")
    pool = set(load_splits()["pool"])
    samples = [s for s in train_samples if s["id"] in pool and s["label"]["tool_calls"]][:6]
    run = tmp_path_factory.mktemp("run")
    with open(run / "trace.jsonl", "w", encoding="utf-8") as t, open(run / "submission.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            env = zaic_tools.ToolEnv("train", s["id"], root=root)
            calls = [dict(c, result=env.call(c["name"], c["arguments"])) for c in s["label"]["tool_calls"]]
            t.write(json.dumps({"id": s["id"], "request": s["audio_text"], "calls": calls}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"id": s["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()},
                               ensure_ascii=False) + "\n")
    evaluate(run / "submission.jsonl", "train", root)
    return run


def test_identity_rewrite_keeps_every_score(gold_run):
    r = rewrite_run(gold_run, default_data_root(), "id", lambda trace, catalog: None)
    assert r["fired"] == 0 and r["diff"] == 0.0
    assert r["out"] == gold_run / "sim" / "id"


def test_replaying_the_same_calls_keeps_every_score(gold_run):
    own = lambda trace, catalog: [{"name": c["name"], "arguments": c["arguments"]} for c in trace["calls"]]  # noqa: E731
    r = rewrite_run(gold_run, default_data_root(), "same", own)
    assert r["fired"] == 6 and r["diff"] == 0.0


def test_dropping_all_calls_scores_zero_on_act_samples(gold_run):
    r = rewrite_run(gold_run, default_data_root(), "drop", lambda trace, catalog: [], out_dir=gold_run / "elsewhere")
    assert r["fired"] == 6 and r["mean_b"] == 0.0
    assert r["out"] == gold_run / "elsewhere" / "drop"
    assert (r["out"] / "per_sample.jsonl").exists()
