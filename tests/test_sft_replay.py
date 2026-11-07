import json

import pytest

from conftest import replay_sample
from lifemate.agents.fewshot import gold_apps
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


def test_records_cli_date_hints_put_the_table_in_the_system_prompt(train_samples, tmp_path):
    from lifemate.agents.dates import date_table
    out = records.main(["--name", "t", "--subset", "pool", "--limit", "3", "--examples", "0", "--tokenizer", "none",
                        "--date-hints", "--out-dir", str(tmp_path)])
    s = {x["id"]: x for x in train_samples}
    for row in map(json.loads, out.read_text(encoding="utf-8").splitlines()):
        assert row["messages"][0]["content"].endswith("\n\n" + date_table(s[row["id"]]["context"]["current_time"]))


def _write_cache(root, name, samples, text_of):
    from lifemate.asr.cached import cache_path, sample_clips

    path = cache_path("train", name)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"audio": clip, "text": text_of(clip), "nbest": []} for s in samples for clip in sample_clips(s)]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _first_pool(train_samples, n):
    pool = set(load_splits()["pool"])
    return [s for s in train_samples if s["id"] in pool][:n]


def test_records_cli_takes_the_request_text_from_an_asr_cache(train_samples, tmp_path, monkeypatch):
    monkeypatch.setenv("LIFEMATE_ASR_CACHE", str(tmp_path / "cache"))
    first = _first_pool(train_samples, 2)
    _write_cache(tmp_path, "noisy", first, lambda clip: f"nghe nhầm {clip}")
    out = records.main(["--name", "t", "--subset", "pool", "--limit", "2", "--examples", "0", "--text", "cache:noisy",
                        "--tokenizer", "none", "--out-dir", str(tmp_path)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["source"] for r in rows] == ["asr", "asr"]
    for s, r in zip(first, rows):
        users = [m["content"] for m in r["messages"] if m["role"] == "user"]
        assert users[-1] == f"nghe nhầm {s['audio']}"
        assert all(u.startswith("nghe nhầm ") for u in users)
        calls = [(c["name"], c["arguments"]) for c in _calls(r)]
        gold = [(g["name"], g["arguments"]) for g in s["label"]["tool_calls"]]
        assert calls[len(calls) - len(gold):] == gold  # history calls first, then the unchanged gold calls


def test_records_cli_raises_on_a_clip_missing_from_the_cache(train_samples, tmp_path, monkeypatch):
    monkeypatch.setenv("LIFEMATE_ASR_CACHE", str(tmp_path / "cache"))
    first = _first_pool(train_samples, 2)
    _write_cache(tmp_path, "partial", first[:1], lambda clip: "x")
    with pytest.raises(KeyError, match=first[1]["audio"]):
        records.main(["--name", "t", "--subset", "pool", "--limit", "2", "--examples", "0", "--text", "cache:partial",
                      "--tokenizer", "none", "--out-dir", str(tmp_path)])


def test_records_cli_writes_gold_masked_and_asr_copies_in_one_file(train_samples, tmp_path, monkeypatch):
    monkeypatch.setenv("LIFEMATE_ASR_CACHE", str(tmp_path / "cache"))
    _write_cache(tmp_path, "noisy", _first_pool(train_samples, 3), lambda clip: "y")
    out = records.main(["--name", "t", "--subset", "pool", "--limit", "3", "--examples", "0", "--text", "gold",
                        "--text", "cache:noisy", "--mask", "1", "--tokenizer", "none", "--out-dir", str(tmp_path)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["source"] for r in rows] == ["gold", "masked", "asr"] * 3


@pytest.mark.parametrize("text", ["whisper", "cache:"])
def test_records_cli_rejects_unknown_text_sources(train_samples, tmp_path, text):
    with pytest.raises(SystemExit):
        records.main(["--name", "t", "--subset", "pool", "--tokenizer", "none", "--out-dir", str(tmp_path),
                      "--text", text])


def test_replay_llm_puts_each_sentence_before_its_call():
    calls = [{"name": "a.b", "arguments": {}}, {"name": "c.d", "arguments": {}}]
    llm = ReplayLLM(calls, "Đã xong.", ["Câu một.", "Câu hai."])
    first = llm.generate([], [])
    assert first.startswith("Câu một.\n<tool_call>") and first.endswith("</tool_call>")
    assert llm.generate([], []).startswith("Câu hai.\n<tool_call>")
    assert llm.generate([], []) == "Đã xong."  # the closing text has no sentence


def test_replay_llm_puts_the_sentence_before_a_refusal():
    assert ReplayLLM([], REFUSAL_TEXT, ["Người dùng tự huỷ yêu cầu."]).generate([], []) == (
        "Người dùng tự huỷ yêu cầu.\n" + REFUSAL_TEXT)


def _steps(sample):
    gold = sample["label"]["tool_calls"]
    return [{"call": c["name"], "thought": f"Câu {i}."} for i, c in enumerate(gold)] or [{"call": None, "thought": "Câu 0."}]


@pytest.mark.parametrize("kind", ["act", "ask", "empty"])
def test_replay_with_thoughts_keeps_the_calls_and_adds_the_sentences(train_samples, kind):
    s = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == kind and not x["history"])
    plain, rec = replay_sample(s), replay_sample(s, thoughts=_steps(s))
    assert _calls(rec) == _calls(plain)
    agent = [m for m in rec["messages"] if m["role"] == "assistant"]
    if kind == "empty":
        assert agent[-1]["content"] == "Câu 0.\n" + REFUSAL_TEXT
    else:
        assert [m["content"] for m in agent if m.get("tool_calls")] == [x["thought"] for x in _steps(s)]


def test_replay_raises_when_the_steps_do_not_name_the_gold_calls(train_samples):
    s = next(x for x in train_samples if label_kind(x["label"]["tool_calls"]) == "act")
    with pytest.raises(ValueError, match=s["id"]):
        replay_sample(s, thoughts=[{"call": "other.tool", "thought": "x"}])


# --- leave-app-out records ---------------------------------------------------------------------------------------------

def _example_tools(rec):
    lines = rec["messages"][0]["content"].splitlines()
    return [json.loads(l[len("<tool_call>"):-len("</tool_call>")])["name"] for l in lines if l.startswith("<tool_call>")]


def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _lao(tmp_path, p, name="t", limit="40"):
    return _read(records.main(["--name", name, "--subset", "pool", "--limit", limit, "--tokenizer", "none",
                               "--out-dir", str(tmp_path), "--leave-app-out", p]))


def test_leave_app_out_removes_gold_app_examples(train_samples, tmp_path):
    by_id = {s["id"]: s for s in train_samples}
    rows = _lao(tmp_path, "1")
    assert rows and all(r["leave_app_out"] for r in rows)
    for r in rows:
        apps = gold_apps(by_id[r["id"]]["label"])
        assert not any(t.split(".")[0] in apps for t in _example_tools(r))


def test_leave_app_out_zero_keeps_the_same_messages(train_samples, tmp_path):
    plain = _read(records.main(["--name", "a", "--subset", "pool", "--limit", "10", "--tokenizer", "none",
                                "--out-dir", str(tmp_path)]))
    zero = _lao(tmp_path, "0", name="b", limit="10")
    assert [r["messages"] for r in plain] == [r["messages"] for r in zero]
    assert not any(r["leave_app_out"] for r in zero)


def test_records_without_the_option_have_no_leave_app_out_key(train_samples, tmp_path):
    plain = _read(records.main(["--name", "a", "--subset", "pool", "--limit", "5", "--tokenizer", "none",
                                "--out-dir", str(tmp_path)]))
    assert plain and not any("leave_app_out" in r for r in plain)  # byte-identical to before the option existed


def test_leave_app_out_share_is_deterministic(train_samples, tmp_path):
    first, again = _lao(tmp_path, "0.5", name="a"), _lao(tmp_path, "0.5", name="b")
    assert first == again and 0 < sum(r["leave_app_out"] for r in first) < len(first)


def test_leave_app_out_keeps_a_sample_with_few_other_apps(train_samples):
    fewest = min((s for s in train_samples if s["label"]["tool_calls"] and s["id"] in set(load_splits()["pool"])),
                 key=lambda s: len(s["context"]["installed_apps"]))
    assert replay_sample(fewest, examples=4, leave_gold_apps_out=True) is not None


def test_leave_app_out_rejects_a_share_outside_zero_to_one(train_samples, tmp_path):
    with pytest.raises(SystemExit):
        _lao(tmp_path, "1.5")
    with pytest.raises(SystemExit):
        _lao(tmp_path, "-0.1")


def test_masked_copies_inherit_the_flag(train_samples, tmp_path):
    rows = _read(records.main(["--name", "m", "--subset", "pool", "--limit", "5", "--tokenizer", "none", "--mask", "1",
                               "--out-dir", str(tmp_path), "--leave-app-out", "1"]))
    assert any(r["source"] == "masked" for r in rows) and all(r["leave_app_out"] for r in rows)
