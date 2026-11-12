import json

import pytest

from lifemate.data import default_data_root, import_zaic_tools
from lifemate.evaluate import evaluate
from lifemate.sim import id_fix, main, rewrite_run, run_vote
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


# --- run_vote: majority over the write and ask calls of several runs ---------------------------------------------------

VOTE_CATALOG = {"m.send": {"kind": "write", "parameters": {"properties": {"to": {}, "body": {"x-match": "fuzzy"}}}},
                "m.find": {"kind": "read", "parameters": {"properties": {"q": {}}}}}


def _vote_trace(sample_id, *calls):
    return {"id": sample_id, "request": "x",
            "calls": [{"name": n, "arguments": a, "result": {}} for n, a in calls]}


def _vote_run(root, name, *traces):
    run = root / name
    run.mkdir()
    (run / "trace.jsonl").write_text("\n".join(json.dumps(t) for t in traces) + "\n", encoding="utf-8")
    return run


def _send(to, body="hi"):
    return "m.send", {"to": to, "body": body}


def test_run_vote_takes_the_majority(tmp_path):
    a = _vote_run(tmp_path, "a", _vote_trace("s1", ("m.find", {"q": "bob"}), _send("b", "other words")))
    b = _vote_run(tmp_path, "b", _vote_trace("s1", _send("b", "hello")))
    first = _vote_trace("s1", _send("a"))
    calls = run_vote(others=[str(a), str(b)])(first, VOTE_CATALOG)
    assert calls == [{"name": "m.find", "arguments": {"q": "bob"}},
                     {"name": "m.send", "arguments": {"to": "b", "body": "other words"}}]  # the first run in the majority


def test_run_vote_keeps_first_without_majority(tmp_path):
    a = _vote_run(tmp_path, "a", _vote_trace("s1", _send("b")))
    b = _vote_run(tmp_path, "b", _vote_trace("s1", _send("c")))
    assert run_vote(others=[str(a), str(b)])(_vote_trace("s1", _send("a")), VOTE_CATALOG) is None


def test_run_vote_two_runs_that_disagree_keep_the_first(tmp_path):
    a = _vote_run(tmp_path, "a", _vote_trace("s1", _send("b")))
    assert run_vote(others=[str(a)])(_vote_trace("s1", _send("a")), VOTE_CATALOG) is None


def test_run_vote_ignores_runs_without_a_write_or_ask(tmp_path):
    a = _vote_run(tmp_path, "a", _vote_trace("s1", ("m.find", {"q": "x"})))
    b = _vote_run(tmp_path, "b", _vote_trace("s1", ("m.find", {"q": "x"})))
    assert run_vote(others=[str(a), str(b)])(_vote_trace("s1", _send("a")), VOTE_CATALOG) is None


def test_run_vote_counts_an_ask_by_tool_and_missing_params(tmp_path):
    ask = lambda params: ("sys.ask_missing_info", {"tool_name": "m.send", "params": params})  # noqa: E731
    a = _vote_run(tmp_path, "a", _vote_trace("s1", ask(["to", "body"])))
    b = _vote_run(tmp_path, "b", _vote_trace("s1", ask(["body", "to"])))
    calls = run_vote(others=[str(a), str(b)])(_vote_trace("s1", _send("a")), VOTE_CATALOG)
    assert calls and calls[0]["name"] == "sys.ask_missing_info"


def test_run_vote_skips_a_run_that_lacks_the_sample(tmp_path):
    a = _vote_run(tmp_path, "a", _vote_trace("other", _send("b")))
    b = _vote_run(tmp_path, "b", _vote_trace("s1", _send("b")))
    assert run_vote(others=[str(a), str(b)])(_vote_trace("s1", _send("a")), VOTE_CATALOG) is None


def test_run_vote_cli_label_is_a_safe_folder_name(gold_run, tmp_path):
    r = main(["run_vote", str(gold_run), "--kw", f"others={json.dumps([str(gold_run), str(gold_run)])}"])
    assert r["fired"] == 0 and r["diff"] == 0.0  # identical runs: the first stands
    assert r["out"].parent == gold_run / "sim" and "/" not in r["out"].name and "\\" not in r["out"].name


# --- date_fix --------------------------------------------------------------------------------------------------------

def test_date_fix_rewrite_changes_only_the_date(tmp_path):
    from lifemate.sim import REWRITES
    catalog = {"calendar.create_event": {"parameters": {"properties": {"start": {"type": "string"}}}}}
    rewrite = REWRITES["date_fix"](times={"s1": "2026-10-01T09:00:00+07:00"})
    trace = {"id": "s1", "request": "họp chiều mai",
             "calls": [{"name": "calendar.create_event", "arguments": {"start": "2026-09-30T15:00:00+07:00"}, "result": {}}]}
    assert rewrite(trace, catalog) == [{"name": "calendar.create_event", "arguments": {"start": "2026-10-02T15:00:00+07:00"}}]
    assert rewrite(dict(trace, request="họp lúc ba giờ"), catalog) is None
    assert rewrite(dict(trace, id="unknown"), catalog) is None



def test_weekday_fix_rewrite_puts_the_weekday_date_into_each_call():
    from lifemate.sim import REWRITES
    catalog = {"tasks.add": {"parameters": {"properties": {"due": {"type": "string"}}}}}
    rewrite = REWRITES["weekday_fix"](times={"s1": "2026-10-01T09:00:00+07:00"})
    trace = {"id": "s1", "request": "tạo lịch Bàn giao lúc mười giờ sáng thứ tư tuần sau",
             "calls": [{"name": "contacts.search", "arguments": {}, "result": {}},
                       {"name": "tasks.add", "arguments": {"due": "2026-10-08T10:00:00+07:00"}, "result": {}}]}
    assert rewrite(trace, catalog) == [{"name": "contacts.search", "arguments": {}},
                                       {"name": "tasks.add", "arguments": {"due": "2026-10-07T10:00:00+07:00"}}]


def test_weekday_fix_rewrite_returns_none_without_a_weekday_phrase():
    from lifemate.sim import REWRITES
    rewrite = REWRITES["weekday_fix"](times={"s1": "2026-10-01T09:00:00+07:00"})
    trace = {"id": "s1", "request": "họp chiều mai",
             "calls": [{"name": "calendar.create_event", "arguments": {"start": "2026-09-30T15:00:00+07:00"}, "result": {}}]}
    assert rewrite(trace, {}) is None
    assert rewrite(dict(trace, id="unknown", request="họp thứ tư tuần sau"), {}) is None

def test_clock_fix_rewrite_fixes_the_first_dated_call_only():
    from lifemate.sim import REWRITES
    rewrite = REWRITES["clock_fix"](times={"s1": "2026-11-05T11:40:00+07:00"})
    trace = {"id": "s1", "request": "Nhớ nhắc mình gọi lại cho khách lúc tám rưỡi tối mai nhé.",
             "calls": [{"name": "contacts.search", "arguments": {}, "result": {}},
                       {"name": "reminders.create", "arguments": {"time": "2026-11-06T19:30:00+07:00"}, "result": {}},
                       {"name": "alarm.create", "arguments": {"time": "2026-11-06T19:00:00+07:00"}, "result": {}}]}
    assert rewrite(trace, {}) == [{"name": "contacts.search", "arguments": {}},
                                  {"name": "reminders.create", "arguments": {"time": "2026-11-06T20:30:00+07:00"}},
                                  {"name": "alarm.create", "arguments": {"time": "2026-11-06T19:00:00+07:00"}}]
    assert rewrite(dict(trace, request="nhắc mình gọi lại cho khách"), {}) is None
    assert rewrite(dict(trace, id="unknown"), {}) is None


ID_CATALOG = {"loyalty.redeem": {"name": "loyalty.redeem", "kind": "write",
                                 "parameters": {"properties": {"program": {"type": "string"}, "reward_id": {"type": "string"}}}}}
REDEEM_BY_NAME = {"name": "loyalty.redeem", "arguments": {"program": "MinhPoint", "reward_id": "rw_1"}, "result": {"ok": True}}


def test_id_fix_rewrite_uses_earlier_results_and_the_history():
    read = {"name": "loyalty.get_points", "arguments": {}, "result": {"programs": [{"id": "prog_9", "name": "MinhPoint"}]}}
    rewrite = id_fix(histories={"s1": [], "s2": [{"role": "assistant", "tool_results": [{"id": "prog_7", "name": "MinhPoint"}]}]})
    assert rewrite({"id": "s1", "calls": [read, REDEEM_BY_NAME]}, ID_CATALOG) == [
        {"name": "loyalty.get_points", "arguments": {}},
        {"name": "loyalty.redeem", "arguments": {"program": "prog_9", "reward_id": "rw_1"}}]
    assert rewrite({"id": "s2", "calls": [REDEEM_BY_NAME]}, ID_CATALOG) == [
        {"name": "loyalty.redeem", "arguments": {"program": "prog_7", "reward_id": "rw_1"}}]


def test_id_fix_rewrite_returns_none_when_nothing_changes():
    rewrite = id_fix(histories={"s1": []})
    assert rewrite({"id": "s1", "calls": [REDEEM_BY_NAME]}, ID_CATALOG) is None  # no entity named MinhPoint seen


def test_main_passes_the_selected_data_root_to_factories_that_take_one(tmp_path):
    from lifemate.sim import build_rewrite

    seen = {}

    def with_root(data_root=None, **kw):
        seen.update(data_root=data_root, **kw)
        return lambda trace, catalog: None

    def without_root(flag=0):
        seen.update(flag=flag)
        return lambda trace, catalog: None

    build_rewrite(with_root, {"x": 1}, tmp_path)
    assert seen == {"data_root": tmp_path, "x": 1}
    build_rewrite(without_root, {"flag": 2}, tmp_path)
    assert seen["flag"] == 2
    build_rewrite(with_root, {"data_root": "explicit"}, tmp_path)  # an explicit --kw wins
    assert seen["data_root"] == "explicit"
