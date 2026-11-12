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
    (("answer", None), ("read", "alarm.list"), None),
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
