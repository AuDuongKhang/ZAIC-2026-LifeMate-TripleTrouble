import json

import pytest

from conftest import replay_sample
from lifemate.agents.prompting import system_prompt
from lifemate.evaluate import label_kind
from lifemate.sft import records
from lifemate.sft.replay import FINAL_TEXT, REFUSAL_TEXT, ReplayLLM
from lifemate.splits import load_splits


def _calls(rec):
    return [c["function"] for m in rec["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])]


def test_replay_llm_returns_gold_calls_then_the_final_text():
    llm = ReplayLLM([{"name": "alarm.list", "arguments": {}}], "Đã xong.")
    first = llm.generate([{"role": "user", "content": "x"}], [])
    assert first.startswith("<tool_call>") and json.loads(first[len("<tool_call>"):-len("</tool_call>")]) == {
        "name": "alarm.list", "arguments": {}}
    assert llm.generate([], []) == "Đã xong."
    assert len(llm.recorded) == 2 and llm.count_tokens([], []) == 1


def test_replay_reproduces_gold_calls_for_every_kind(train_samples):
    for kind in ("act", "ask", "empty"):
        s = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == kind)
        rec = replay_sample(s)
        assert rec is not None
        assert [(c["name"], c["arguments"]) for c in _calls(rec)] == [
            (g["name"], g["arguments"]) for g in s["label"]["tool_calls"]]


def test_replay_messages_match_what_the_agent_sends(train_samples):
    s = next(x for x in train_samples if len(x["label"]["tool_calls"]) >= 2 and not x["history"])
    rec = replay_sample(s)
    assert rec["messages"][0]["role"] == "system" and rec["messages"][0]["content"] == system_prompt(s["context"], "v1")
    assert [m["role"] for m in rec["messages"]].count("tool") == len(s["label"]["tool_calls"])
    assert rec["messages"][1] == {"role": "user", "content": s["audio_text"]}


def test_ask_ends_on_the_ask_call_and_act_on_final_text(train_samples):
    ask = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "ask")
    act = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "act")
    empty = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "empty")
    assert replay_sample(ask)["messages"][-1]["tool_calls"][0]["function"]["name"] == "sys.ask_missing_info"
    assert replay_sample(act)["messages"][-1] == {"role": "assistant", "content": FINAL_TEXT}
    assert replay_sample(empty)["messages"][-1] == {"role": "assistant", "content": REFUSAL_TEXT}


def test_replay_with_examples_puts_them_in_the_system_prompt(train_samples):
    s = next(x for x in train_samples if x["label"]["tool_calls"])
    rec = replay_sample(s, examples=2)
    assert "Examples of correct behaviour" in rec["messages"][0]["content"]


def test_replay_returns_none_when_calls_differ(train_samples):
    s = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "act")
    broken = dict(s, label=dict(s["label"], tool_calls=[{"name": "no_such.tool", "arguments": {}}]))
    assert replay_sample(broken) is None


def test_records_cli_uses_pool_only(train_samples, tmp_path):
    out = records.main(["--name", "t", "--subset", "pool", "--limit", "30", "--examples", "0", "--tokenizer", "none",
                        "--out-dir", str(tmp_path)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    pool = set(load_splits()["pool"])
    assert len(rows) == 30 and all(r["id"] in pool and r["source"] == "gold" for r in rows)


@pytest.mark.parametrize("flag", [["--text", "cache:x"]])
def test_records_cli_rejects_later_options(train_samples, tmp_path, flag):
    with pytest.raises(SystemExit):
        records.main(["--name", "t", "--subset", "pool", "--tokenizer", "none", "--out-dir", str(tmp_path), *flag])
