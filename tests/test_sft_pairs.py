import pytest

from lifemate.sft.pairs import decision, pair_type

KINDS = {"alarm.create": "write", "alarm.list": "read", "contacts.search": "read", "sms.send": "write"}
MAP = {"orbit.make": "alarm.create", "orbit.find": "alarm.list", "reef.lookup": "contacts.search", "comet.dispatch": "sms.send"}


def msg(name=None, **arguments):
    if name is None:
        return {"role": "assistant", "content": "Đã xong."}
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def test_decision_maps_names_back_and_reads_the_kind():
    assert decision(msg("orbit.make", t="x"), MAP, KINDS) == ("write", "alarm.create")
    assert decision(msg("reef.lookup"), MAP, KINDS) == ("read", "contacts.search")
    assert decision(msg("sys.ask_missing_info", tool_name="comet.dispatch", params=["to"]), MAP, KINDS) == ("ask", "sms.send")
    assert decision(msg(), MAP, KINDS) == ("answer", None)
    assert decision(msg("nebula.invent"), MAP, KINDS) == ("other", None)


@pytest.mark.parametrize("gold, model, expected", [
    (("write", "alarm.create"), ("write", "alarm.create"), None),
    (("write", "alarm.create"), ("write", "sms.send"), "write:other_write"),
    (("write", "alarm.create"), ("read", "alarm.list"), "write:read"),
    (("write", "alarm.create"), ("ask", "alarm.create"), "write:ask"),
    (("write", "alarm.create"), ("answer", None), "write:answer"),
    (("write", "alarm.create"), ("other", None), None),
    (("ask", "sms.send"), ("ask", "alarm.create"), None),
    (("ask", "sms.send"), ("write", "sms.send"), "ask:write"),
    (("ask", "sms.send"), ("read", "contacts.search"), "ask:read"),
    (("ask", "sms.send"), ("answer", None), "ask:answer"),
    (("answer", None), ("read", "alarm.list"), "answer:read"),
    (("answer", None), ("write", "alarm.create"), "answer:write"),
    (("answer", None), ("ask", "sms.send"), "answer:ask"),
    (("read", "alarm.list"), ("read", "contacts.search"), None),
    (("read", "alarm.list"), ("write", "alarm.create"), "read:write"),
    (("read", "alarm.list"), ("ask", "sms.send"), "read:ask"),
    (("read", "alarm.list"), ("answer", None), "read:answer"),
    (("read", "alarm.list"), ("other", None), None),
])
def test_pairing_table(gold, model, expected):
    assert pair_type(gold, model) == expected


def test_an_unparsable_output_is_other():
    from lifemate.sft.replay import assistant_message
    broken = assistant_message('<tool_call>{"name": "orbit.make", "arguments": {</tool_call>')
    assert decision(broken, MAP, KINDS) == ("other", None)


FUZZY = {"alarm.create": {"label"}, "sms.send": {"content"}}
MAP_P = MAP | {"stamp": "time", "tag": "label"}


@pytest.mark.parametrize("gold_args, model_args, expected", [
    ({"time": "2026-11-06T20:30:00+07:00"}, {"stamp": "2026-11-06T19:30:00+07:00"}, True),   # non-fuzzy value differs
    ({"time": "2026-11-06T20:30:00+07:00"}, {"stamp": "2026-11-06T20:30:00+07:00"}, False),
    ({"time": "t", "label": "Đi chợ"}, {"stamp": "t", "tag": "đi chợ sáng"}, False),         # fuzzy key ignored
    ({"time": "t"}, {"stamp": "t", "tag": "x"}, False),                                        # optional key on one side only
    ({"time": "t", "repeat": "daily"}, {"stamp": "t"}, False),
])
def test_args_differ_on_non_fuzzy_keys_only(gold_args, model_args, expected):
    from lifemate.sft.pairs import args_differ
    gold = msg("alarm.create", **gold_args)
    model = msg("orbit.make", **model_args)
    assert args_differ(gold, model, MAP_P, FUZZY) is expected


def test_args_differ_needs_the_same_write_tool():
    from lifemate.sft.pairs import args_differ
    assert args_differ(msg("alarm.create", time="a"), msg("comet.dispatch", time="b"), MAP_P, FUZZY) is False


def test_tool_kinds_reads_the_catalog_without_the_ask_tool():
    from lifemate.sft.pairs import tool_kinds
    catalog = {"a.x": {"kind": "read"}, "a.y": {"kind": "write"}, "sys.ask_missing_info": {"kind": "write"}}
    assert tool_kinds(catalog) == {"a.x": "read", "a.y": "write"}


# --- Task 2: choosing the rejected step, pair files ------------------------------------------------------------------

import json  # noqa: E402

from lifemate.sft.replay import assistant_message  # noqa: E402

FUZZY_A = {"alarm.create": {"label"}}


def call_text(name, **arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "</tool_call>"


GOLD_WRITE = msg("alarm.create", time="2026-11-06T20:30:00+07:00", label="Đi chợ")


def test_choose_rejected_takes_the_most_frequent_wrong_decision():
    from lifemate.sft.pairs import choose_rejected
    read = call_text("alarm.list")
    candidates = [call_text("alarm.create", time="2026-11-06T20:30:00+07:00", label="Đi chợ"), read, "Đã xong.", read]
    assert choose_rejected(GOLD_WRITE, candidates, {}, KINDS, FUZZY_A) == ("write:read", read, False)


def test_choose_rejected_breaks_ties_by_order_and_flags_a_wrong_greedy():
    from lifemate.sft.pairs import choose_rejected
    candidates = ["Đã xong.", call_text("alarm.list")]
    assert choose_rejected(GOLD_WRITE, candidates, {}, KINDS, FUZZY_A) == ("write:answer", "Đã xong.", True)


def test_no_pair_when_only_fuzzy_arguments_differ():
    from lifemate.sft.pairs import choose_rejected
    candidates = [call_text("alarm.create", time="2026-11-06T20:30:00+07:00", label="đi chợ sáng")] * 3
    assert choose_rejected(GOLD_WRITE, candidates, {}, KINDS, FUZZY_A) is None


def test_write_args_pair_when_a_non_fuzzy_argument_differs():
    from lifemate.sft.pairs import choose_rejected
    other = call_text("alarm.create", time="2026-11-06T19:30:00+07:00", label="Đi chợ")
    candidates = [call_text("alarm.create", time="2026-11-06T20:30:00+07:00", label="Đi chợ"), other, other]
    assert choose_rejected(GOLD_WRITE, candidates, {}, KINDS, FUZZY_A) == ("write:args", other, False)


def _record(sample_id):
    """A masked record with two agent turns: write orbit.make (alarm.create), then the closing text."""
    gold = {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "orbit.make", "arguments": {"stamp": "2026-11-06T20:30:00+07:00"}}}]}
    return {"id": sample_id, "source": "masked", "name_map": {"orbit.make": "alarm.create", "stamp": "time"},
            "tools": [{"type": "function", "function": {"name": "orbit.make", "parameters": {"properties": {"stamp": {}}}}}],
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "đặt báo thức tám rưỡi tối"},
                         gold, {"role": "tool", "content": "{\"ok\": true}"},
                         {"role": "assistant", "content": "Đã xong."}]}


def _write_inputs(tmp_path, ids):
    (tmp_path / "steps").mkdir(parents=True)
    records = [_record(i) for i in ids]
    (tmp_path / "rec.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    rows = []
    for k, i in enumerate(ids):
        base = {"records": "rec", "record": k, "id": i, "temperature": 0.8, "top_p": 0.95, "seed": 0, "model": "m",
                "revision": "r"}
        rows.append(dict(base, turn=2, greedy="Đã xong.", samples=[call_text("orbit.make", stamp="2026-11-06T20:30:00+07:00")]))
        rows.append(dict(base, turn=4, greedy="Đã xong.", samples=["Đã xong."]))
    (tmp_path / "steps" / "st.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                 encoding="utf-8")


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_writes_train_heldout_review_and_report(tmp_path):
    from lifemate.sft import pairs
    _write_inputs(tmp_path, ["train_0001", "train_0009"])  # train_0009 is held out (crc32 % 10 == 0)
    assert pairs.is_heldout("train_0009") and not pairs.is_heldout("train_0001")
    report = pairs.main(["--records", "rec", "--steps", "st", "--out", "p", "--records-dir", str(tmp_path)])
    assert report["points"] == 4 and report["with_pair"] == 2 and report["rate"] == 0.5
    assert report["greedy_wrong"] == 2 and report["by_type"] == {"write:answer": 2}
    train, held = _read(tmp_path / "pairs" / "p.jsonl"), _read(tmp_path / "pairs" / "p-heldout.jsonl")
    assert [p["id"] for p in train] == ["train_0001"] and [p["id"] for p in held] == ["train_0009"]
    pair = train[0]
    assert pair["turn"] == 2 and pair["messages"][-1]["role"] == "user"
    assert pair["chosen"] == _record("train_0001")["messages"][2]
    assert pair["rejected"] == assistant_message("Đã xong.") and pair["type"] == "write:answer"
    assert (tmp_path / "pairs" / "p-review.md").exists()


def test_greedy_wrong_only_keeps_only_those_pairs(tmp_path):
    from lifemate.sft import pairs
    _write_inputs(tmp_path, ["train_0001"])
    rows = _read(tmp_path / "steps" / "st.jsonl")
    rows[0].update(greedy=call_text("orbit.make", stamp="2026-11-06T20:30:00+07:00"), samples=["Đã xong."])
    (tmp_path / "steps" / "st.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                 encoding="utf-8")
    report = pairs.main(["--records", "rec", "--steps", "st", "--out", "p", "--records-dir", str(tmp_path),
                         "--greedy-wrong-only"])
    assert report["with_pair"] == 1 and report["train"] == 0
    assert _read(tmp_path / "pairs" / "p.jsonl") == []
