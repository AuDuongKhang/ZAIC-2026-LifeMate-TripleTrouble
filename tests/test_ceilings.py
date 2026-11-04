import json

import pytest

from lifemate.ceilings import BUCKETS, arg_type, fix_rewrite, main
from lifemate.data import default_data_root, import_zaic_tools, load_catalog
from lifemate.evaluate import evaluate
from lifemate.sim import rewrite_run
from lifemate.splits import load_splits

MAY_REORDER_WRITES = {"write_set", "everything"}


@pytest.mark.parametrize("spec, value, expected", [
    ({"x-match": "fuzzy", "x-ref": "contacts", "type": "string"}, "2026-11-14", "free"),
    ({"x-ref": "contacts", "type": "string"}, "2026-11-14", "id"),
    ({"type": "array", "items": {"type": "string", "x-ref": "contacts"}}, ["c1"], "id"),
    ({"type": "string", "format": "date"}, "ngày mai", "date"),
    ({"type": "string", "format": "time"}, "sáng", "date"),
    ({"type": "string"}, "09:30", "date"),
    ({"type": "string", "enum": ["2026-11-14", "x"]}, "2026-11-14", "date"),
    ({"type": "boolean"}, True, "enum_bool"),
    ({"type": "string", "enum": ["a", "b"]}, "a", "enum_bool"),
    ({"type": "integer", "enum": [1, 2]}, 1, "enum_bool"),
    ({"type": "integer"}, 3, "num"),
    ({"type": "number"}, 1.5, "num"),
    ({"type": "string"}, "xin chào", "string"),
])
def test_arg_type_checks_the_rules_in_order(spec, value, expected):
    assert arg_type({"parameters": {"properties": {"k": spec}}}, "k", value) == expected


def test_arg_type_reads_the_value_when_the_schema_has_no_format():
    spec = {"parameters": {"properties": {"when": {"type": "string"}}}}
    assert arg_type(spec, "when", "2026-11-14") == "date"


def test_arg_type_reads_the_value_when_the_tool_or_key_is_unknown():
    assert arg_type(None, "when", "2026-11-14") == "date"
    assert arg_type(None, "count", 3) == "num"
    assert arg_type({"parameters": {"properties": {}}}, "flag", True) == "enum_bool"
    assert arg_type(None, "x", "abc") == "string"


def test_buckets_are_the_planned_set():
    assert set(BUCKETS) == {
        "free_text", "id", "date", "num", "string", "enum_bool", "all_args", "reads", "reads_drop_extra",
        "reads_add_missing", "reads_fix_args", "ask_act", "ask_gold_asks", "ask_agent_asks", "write_set", "out_of_scope",
        "all_args+reads", "all_args+reads+ask", "everything"}


def test_each_bucket_fixes_what_its_name_says():
    every_type = {"free", "id", "date", "num", "string", "enum_bool"}
    assert BUCKETS == {
        "free_text": {"arg_types": {"free"}},
        "id": {"arg_types": {"id"}},
        "date": {"arg_types": {"date"}},
        "num": {"arg_types": {"num"}},
        "string": {"arg_types": {"string"}},
        "enum_bool": {"arg_types": {"enum_bool"}},
        "all_args": {"arg_types": every_type},
        "reads": {"reads": "all"},
        "reads_drop_extra": {"reads": "drop_extra"},
        "reads_add_missing": {"reads": "add_missing"},
        "reads_fix_args": {"reads": "fix_args"},
        "ask_act": {"ask": "all"},
        "ask_gold_asks": {"ask": "gold_asks"},
        "ask_agent_asks": {"ask": "agent_asks"},
        "write_set": {"write_set": True},
        "out_of_scope": {"out_of_scope": True},
        "all_args+reads": {"arg_types": every_type, "reads": "all"},
        "all_args+reads+ask": {"arg_types": every_type, "reads": "all", "ask": "all"},
        "everything": {"arg_types": every_type, "reads": "all", "ask": "all", "write_set": True, "out_of_scope": True},
    }


def _catalog():
    return {
        "a.get": {"name": "a.get", "kind": "read", "parameters": {"properties": {"q": {"type": "string"}}}},
        "a.set": {"name": "a.set", "kind": "write", "parameters": {"properties": {
            "note": {"type": "string", "x-match": "fuzzy"}, "id": {"type": "string", "x-ref": "a"},
            "day": {"type": "string"}, "n": {"type": "integer"}}}},
        "sys.ask_missing_info": {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"properties": {}}},
    }


def _call(name, **arguments):
    return {"name": name, "arguments": arguments}


def _fix(gold, pred, **kw):
    """Run fix_rewrite on one in-memory sample; the trace line holds results like a real one."""
    trace = {"id": "s", "calls": [dict(c, result={"ok": True}) for c in pred]}
    return fix_rewrite({"s": {"id": "s", "label": {"tool_calls": gold}}}, **kw)(trace, _catalog())


def test_arg_types_fix_only_the_named_type():
    gold = [_call("a.set", note="đúng", id="i1", day="2026-11-14", n=1)]
    pred = [_call("a.set", note="sai", id="i2", day="2026-11-15", n=2)]
    assert _fix(gold, pred, arg_types={"date"}) == [_call("a.set", note="sai", id="i2", day="2026-11-14", n=2)]
    assert _fix(gold, pred, arg_types={"free"}) == [_call("a.set", note="đúng", id="i2", day="2026-11-15", n=2)]


def test_arg_types_take_the_type_from_the_gold_value():
    """A model that writes "thứ ba" where gold has an ISO date made a date error, not a string error."""
    gold, pred = [_call("a.set", day="2026-11-14")], [_call("a.set", day="thứ ba")]
    assert _fix(gold, pred, arg_types={"date"}) == gold
    assert _fix(gold, pred, arg_types={"string"}) is None


def test_arg_types_add_a_missing_argument_and_drop_an_extra_one():
    gold = [_call("a.set", id="i1", n=1)]
    pred = [_call("a.set", id="i1", note="thừa")]
    assert _fix(gold, pred, arg_types={"free"}) == [_call("a.set", id="i1")]
    assert _fix(gold, pred, arg_types={"num"}) == [_call("a.set", id="i1", note="thừa", n=1)]


def test_arg_types_leave_reads_alone():
    assert _fix([_call("a.get", q="2026-11-14")], [_call("a.get", q="2026-11-15")], arg_types={"date"}) is None


def test_alignment_gives_each_pred_call_in_turn_its_best_remaining_gold_call():
    """The plan's numbers (id +1.82, write_set +11.43) come from this order; best overlap over all pairs gives +1.57, +12.72."""
    gold = [_call("a.set", id="i1", n=1), _call("a.set", id="i2", n=2)]
    assert _fix(gold, [_call("a.set", id="i2", n=9)], arg_types={"num"}) == [_call("a.set", id="i2", n=2)]
    pred = [_call("a.set", id="i2", n=9), _call("a.set", id="i2", n=2)]
    assert _fix(gold, pred, arg_types={"num", "id"}) == [_call("a.set", id="i2", n=2), _call("a.set", id="i1", n=1)]


def test_read_modes():
    gold = [_call("a.get", q="x"), _call("a.get", q="y"), _call("a.set", id="i")]
    pred = [_call("a.get", q="x"), _call("a.get", q="y2"), _call("a.get", q="z"), _call("a.set", id="i")]
    two = [_call("a.get", q="x"), _call("a.get", q="y"), _call("a.set", id="i")]
    assert _fix(gold, pred, reads="all") == two
    assert _fix(gold, pred, reads="fix_args") == [_call("a.get", q="x"), _call("a.get", q="y"), _call("a.get", q="z"),
                                                  _call("a.set", id="i")]
    assert _fix(gold, pred, reads="drop_extra") == [_call("a.get", q="x"), _call("a.get", q="y2"), _call("a.set", id="i")]
    short = [_call("a.get", q="x"), _call("a.set", id="i")]
    assert _fix(gold, short, reads="add_missing") == [_call("a.get", q="x"), _call("a.get", q="y"), _call("a.set", id="i")]


def test_ask_modes_replace_the_whole_sample_with_gold():
    ask = _call("sys.ask_missing_info", tool_name="a.set", params=["id"])
    write = _call("a.set", id="i")
    assert _fix([ask], [write], ask="gold_asks") == [ask]
    assert _fix([ask], [write], ask="agent_asks") is None
    assert _fix([ask], [write], ask="all") == [ask]
    assert _fix([write], [ask], ask="agent_asks") == [write]
    assert _fix([write], [ask], ask="gold_asks") is None
    assert _fix([write], [ask], ask="all") == [write]


def test_write_set_adds_missing_writes_and_drops_extra_ones():
    one, two = _call("a.set", id="i1"), _call("a.set", id="i2")
    assert _fix([one, two], [one], write_set=True) == [one, two]
    assert _fix([one], [one, two], write_set=True) == [one]


def test_out_of_scope_flag_only_touches_gold_empty_samples():
    write = _call("a.set", id="i")
    assert _fix([], [write], out_of_scope=True) == []
    assert _fix([], [], out_of_scope=True) is None
    assert _fix([write], [_call("a.set", id="j")], out_of_scope=True) is None
    assert _fix([], [write], arg_types={"id"}, reads="all", write_set=True) is None


@pytest.fixture(scope="module")
def fake_run(train_samples, tmp_path_factory):
    """A fake pool run: gold samples, an out-of-scope sample with a call, an ask sample with none, and perturbed samples."""
    root = default_data_root()
    zaic_tools, catalog = import_zaic_tools(root, "train"), load_catalog(root, "train")
    pool = set(load_splits()["pool"])
    samples = [s for s in train_samples if s["id"] in pool]
    golds = [s for s in samples if s["label"]["tool_calls"] and not _asks(s)][:3]
    out_of_scope = next(s for s in samples if not s["label"]["tool_calls"])
    asks = next(s for s in samples if _asks(s))
    has_read = [s for s in samples if s not in golds and _of_kind(s, catalog, "read") and not _asks(s)][:3]
    has_write = [s for s in samples if s not in golds and s not in has_read and _of_kind(s, catalog, "write")
                 and not _asks(s)][:3]
    extra_read = next(c for s in samples for c in s["label"]["tool_calls"] if catalog[c["name"]]["kind"] == "read")

    calls = {s["id"]: s["label"]["tool_calls"] for s in golds}
    calls[out_of_scope["id"]] = [extra_read]
    calls[asks["id"]] = []
    for s, how in zip(has_read, ("drop", "extra", "args")):
        gold = s["label"]["tool_calls"]
        read = next(i for i, c in enumerate(gold) if catalog[c["name"]]["kind"] == "read" and c["arguments"])
        if how == "drop":
            calls[s["id"]] = gold[:read] + gold[read + 1:]
        elif how == "extra":
            calls[s["id"]] = gold + [extra_read]
        else:
            calls[s["id"]] = gold[:read] + [dict(gold[read], arguments={})] + gold[read + 1:]
    for s, value in zip(has_write, ("sai", 999, True)):
        gold = s["label"]["tool_calls"]
        write = next(i for i, c in enumerate(gold) if catalog[c["name"]]["kind"] == "write" and c["arguments"])
        wrong = {k: value for k in gold[write]["arguments"]}
        calls[s["id"]] = gold[:write] + [dict(gold[write], arguments=wrong)] + gold[write + 1:]

    run = tmp_path_factory.mktemp("run")
    with open(run / "trace.jsonl", "w", encoding="utf-8") as t, open(run / "submission.jsonl", "w", encoding="utf-8") as f:
        for sid, pred in calls.items():
            env = zaic_tools.ToolEnv("train", sid, root=root)
            traced = [dict(c, result=env.call(c["name"], c["arguments"])) for c in pred]
            t.write(json.dumps({"id": sid, "request": "", "calls": traced}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"id": sid, "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()},
                               ensure_ascii=False) + "\n")
    evaluate(run / "submission.jsonl", "train", root)
    (run / "meta.json").write_text(json.dumps({"split": "train"}), encoding="utf-8")
    return run, {s["id"] for s in golds}, out_of_scope["id"]


def _asks(sample):
    return any(c["name"] == "sys.ask_missing_info" for c in sample["label"]["tool_calls"])


def _of_kind(sample, catalog, kind):
    return any(catalog[c["name"]]["kind"] == kind and c["arguments"] for c in sample["label"]["tool_calls"])


@pytest.fixture(scope="module")
def samples_by_id(train_samples):
    return {s["id"]: s for s in train_samples}


@pytest.mark.parametrize("bucket", sorted(BUCKETS))
def test_fix_rewrite_returns_none_when_nothing_changes(fake_run, samples_by_id, bucket):
    run, golds, _ = fake_run
    catalog = load_catalog(default_data_root(), "train")
    rewrite = fix_rewrite(samples_by_id, **BUCKETS[bucket])
    traces = [json.loads(line) for line in open(run / "trace.jsonl", encoding="utf-8")]
    for trace in traces:
        if trace["id"] in golds:
            assert rewrite(trace, catalog) is None


@pytest.mark.parametrize("bucket", sorted(BUCKETS))
def test_all_buckets_score_at_least_the_run(fake_run, samples_by_id, tmp_path, bucket):
    run, _, _ = fake_run
    r = rewrite_run(run, default_data_root(), bucket, fix_rewrite(samples_by_id, **BUCKETS[bucket]), out_dir=tmp_path)
    if bucket not in MAY_REORDER_WRITES:
        assert r["worse"] == 0
    assert r["diff"] >= 0


def test_the_perturbed_run_gains_from_the_full_fix(fake_run, samples_by_id, tmp_path):
    run, _, _ = fake_run
    r = rewrite_run(run, default_data_root(), "everything", fix_rewrite(samples_by_id, **BUCKETS["everything"]),
                    out_dir=tmp_path)
    assert r["fired"] >= 8 and r["diff"] > 0 and r["mean_b"] == 100


def test_out_of_scope_fix_is_an_empty_call_list(fake_run, samples_by_id, tmp_path):
    run, _, sid = fake_run
    catalog = load_catalog(default_data_root(), "train")
    trace = next(t for t in map(json.loads, open(run / "trace.jsonl", encoding="utf-8")) if t["id"] == sid)
    rewrite = fix_rewrite(samples_by_id, out_of_scope=True)
    assert trace["calls"] and rewrite(trace, catalog) == []
    r = rewrite_run(run, default_data_root(), "out_of_scope", rewrite, out_dir=tmp_path)
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(r["out"] / "per_sample.jsonl", encoding="utf-8")}
    assert rows[sid]["score"] == 1


def test_cli_writes_each_bucket_and_the_table(fake_run):
    run, _, _ = fake_run
    main([str(run)])
    assert all((run / "ceilings" / bucket / "per_sample.jsonl").exists() for bucket in BUCKETS)
    table = (run / "ceilings.md").read_text(encoding="utf-8")
    assert all(f"| {bucket} |" in table for bucket in BUCKETS)


def test_cli_refuses_a_run_that_is_not_on_train(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"split": "public_test"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="train"):
        main([str(tmp_path)])


def test_cli_refuses_a_run_without_meta(tmp_path):
    with pytest.raises(SystemExit, match="meta.json"):
        main([str(tmp_path)])


def test_cli_refuses_an_unscored_run_before_writing_anything(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"split": "train"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="lifemate.evaluate"):
        main([str(tmp_path)])
    assert not (tmp_path / "ceilings").exists()
