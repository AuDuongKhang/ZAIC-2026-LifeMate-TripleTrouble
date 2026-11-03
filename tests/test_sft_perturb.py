import json
import random

import pytest

from conftest import replay_sample
from lifemate.agents.parsing import BLOCK
from lifemate.evaluate import label_kind
from lifemate.sft import records
from lifemate.sft.perturb import perturb_record


def _first_record(train_samples, want, examples=0):
    for s in train_samples:
        if want(s):
            rec = replay_sample(s, examples=examples)
            if rec is not None:
                return rec
    raise AssertionError("no sample fits")


@pytest.fixture(scope="module")
def act_record(train_samples):
    """An act sample with several calls and tool calls in its history."""
    return _first_record(train_samples, lambda s: label_kind(s["label"]["tool_calls"]) == "act"
                         and len(s["label"]["tool_calls"]) >= 2
                         and any(t.get("tool_calls") for t in s["history"]))


@pytest.fixture(scope="module")
def ask_record(train_samples):
    return _first_record(train_samples, lambda s: label_kind(s["label"]["tool_calls"]) == "ask")


@pytest.fixture(scope="module")
def examples_record(train_samples):
    return _first_record(train_samples, lambda s: label_kind(s["label"]["tool_calls"]) == "act", examples=4)


def _props(rec):
    return {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in rec["tools"]}


def _message_calls(rec):
    return [c["function"] for m in rec["messages"] for c in m.get("tool_calls", [])]


def _example_calls(rec):
    return [json.loads(BLOCK.match(line).group(1)) for line in rec["messages"][0]["content"].split("\n")
            if line.startswith("<tool_call>")]


@pytest.mark.parametrize("fixture", ["act_record", "ask_record", "examples_record"])
def test_perturb_keeps_every_used_parameter(fixture, request):
    rec = request.getfixturevalue(fixture)
    for seed in range(5):
        props = _props(perturb_record(rec, random.Random(seed)))
        for call in _message_calls(rec) + _example_calls(rec):
            assert set(call["arguments"]) <= set(props[call["name"]]), call
            if call["name"] == "sys.ask_missing_info":
                assert set(call["arguments"]["params"]) <= set(props[call["arguments"]["tool_name"]])


def test_examples_record_has_few_shot_calls(examples_record):
    assert _example_calls(examples_record)  # otherwise the few-shot half of the test above checks nothing


def test_perturb_keeps_required_and_messages(act_record):
    p = perturb_record(act_record, random.Random(0))
    assert p["messages"] == act_record["messages"]
    req = lambda r: {t["function"]["name"]: set(t["function"]["parameters"].get("required", [])) for t in r["tools"]}
    assert req(p) == req(act_record)
    assert p["perturbed"] is True and "perturbed" not in act_record


def test_perturb_drops_some_unused_optional_parameter(act_record):
    orig = _props(act_record)
    dropped = 0
    for seed in range(10):
        props = _props(perturb_record(act_record, random.Random(seed)))
        dropped += sum(len(orig[name]) - len(props[name]) for name in orig)
    assert dropped > 0


def test_perturb_shuffles_property_order(act_record):
    orders = {tuple(tuple(p) for p in _props(perturb_record(act_record, random.Random(s), drop_p=0.0)).values())
              for s in range(5)}
    assert len(orders) > 1


def test_perturb_is_deterministic(act_record):
    assert perturb_record(act_record, random.Random(3)) == perturb_record(act_record, random.Random(3))


def test_records_cli_perturbs_only_masked_copies(train_samples, tmp_path):
    out = records.main(["--name", "t", "--subset", "pool", "--limit", "3", "--examples", "0", "--mask", "1",
                        "--perturb", "--tokenizer", "none", "--out-dir", str(tmp_path)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["source"] for r in rows] == ["gold", "masked"] * 3
    assert all("perturbed" not in r for r in rows if r["source"] == "gold")
    assert all(r["perturbed"] is True for r in rows if r["source"] == "masked")


def test_perturb_keeps_the_parameters_a_one_of_group_lists():
    """tool_schema turns x-one-of into "Provide one of: ..." with optional parameters; dropping them would leave a
    description that names parameters the schema no longer has (phone.call could not be called at all)."""
    tool = {"type": "function", "function": {
        "name": "phone.call", "description": "Gọi điện Provide one of: contact_id | phone_number + label.",
        "parameters": {"type": "object", "required": [], "properties": {
            "contact_id": {"type": "string"}, "phone_number": {"type": "string"}, "label": {"type": "string"},
            "speaker": {"type": "boolean"}}}}}
    rec = {"id": "x", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
           "tools": [tool]}
    props = perturb_record(rec, random.Random(0), drop_p=1.0)["tools"][0]["function"]["parameters"]["properties"]
    assert set(props) == {"contact_id", "phone_number", "label"}
