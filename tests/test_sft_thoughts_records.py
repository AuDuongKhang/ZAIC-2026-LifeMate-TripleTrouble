"""records --thoughts: reasoning sentences rendered into every copy of a sample, or the sample dropped altogether."""
import json

import pytest

from lifemate.sft import records
from lifemate.sft.replay import FINAL_TEXT, REFUSAL_TEXT
from lifemate.sft.thoughts import load_thoughts
from lifemate.splits import load_splits

from test_sft_replay import _first_pool, _write_cache


def _write_thoughts(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            gold = s["label"]["tool_calls"]
            steps = [{"call": c["name"], "thought": f"Câu {i}."} for i, c in enumerate(gold)] or [{"call": None, "thought": "Câu 0."}]
            f.write(json.dumps({"id": s["id"], "teacher": "t", "revision": "r", "steps": steps}, ensure_ascii=False) + "\n")


def _run(tmp_path, thoughts, *extra, n=None, prompt="v3"):
    args = ["--name", "t", "--subset", "pool", "--examples", "0", "--tokenizer", "none", "--out-dir", str(tmp_path),
            "--prompt", prompt, "--thoughts", str(thoughts), *extra]
    if n is not None:
        args += ["--limit", str(n)]
    out = records.main(args)
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def _check_sentences(rec):
    last_user = max(i for i, m in enumerate(rec["messages"]) if m["role"] == "user")
    agent = [m for m in rec["messages"][last_user + 1:] if m["role"] == "assistant"]
    with_calls = [m for m in agent if m.get("tool_calls")]
    assert [m["content"] for m in with_calls] == [f"Câu {i}." for i in range(len(with_calls))]
    for m in agent:
        if not m.get("tool_calls"):
            assert m["content"] in (FINAL_TEXT, "Câu 0.\n" + REFUSAL_TEXT)


def test_every_copy_carries_the_sentences(train_samples, tmp_path, monkeypatch):
    monkeypatch.setenv("LIFEMATE_ASR_CACHE", str(tmp_path / "cache"))
    first = _first_pool(train_samples, 3)
    _write_cache(tmp_path, "noisy", first, lambda clip: f"nghe nhầm {clip}")
    _write_thoughts(tmp_path / "th.jsonl", first)
    rows = _run(tmp_path, tmp_path / "th.jsonl", "--text", "gold", "--text", "cache:noisy", "--mask", "1", n=3)
    assert [r["source"] for r in rows] == ["gold", "masked", "asr"] * 3
    for r in rows:
        _check_sentences(r)


def test_a_missing_sample_loses_all_its_copies(train_samples, tmp_path, capsys):
    first = _first_pool(train_samples, 30)
    missing_id = first[7]["id"]
    _write_thoughts(tmp_path / "th.jsonl", [s for s in first if s["id"] != missing_id])
    rows = _run(tmp_path, tmp_path / "th.jsonl", "--text", "gold", "--mask", "1", n=30)
    assert missing_id not in {r["id"] for r in rows} and len(rows) == 58
    assert "1 samples without sentences" in capsys.readouterr().out


def test_five_percent_missing_is_allowed(train_samples, tmp_path):
    first = _first_pool(train_samples, 20)
    _write_thoughts(tmp_path / "th.jsonl", first[1:])
    assert len(_run(tmp_path, tmp_path / "th.jsonl", n=20)) == 19


def test_over_five_percent_missing_stops_and_keeps_the_old_file(train_samples, tmp_path):
    first = _first_pool(train_samples, 20)
    _write_thoughts(tmp_path / "th.jsonl", first[2:])
    out = tmp_path / "t.jsonl"
    out.write_text("old", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(tmp_path, tmp_path / "th.jsonl", n=20)
    assert out.read_text(encoding="utf-8") == "old"


def test_other_subsets_in_the_file_stay_out(train_samples, tmp_path):
    pool = _first_pool(train_samples, 5)
    holdout_ids = set(load_splits()["holdout"])
    holdout = [s for s in train_samples if s["id"] in holdout_ids][:5]
    _write_thoughts(tmp_path / "th.jsonl", pool + holdout)
    rows = _run(tmp_path, tmp_path / "th.jsonl", n=5)
    assert len(rows) == 5 and {r["id"] for r in rows} <= set(load_splits()["pool"])


def test_a_wrong_call_name_raises(train_samples, tmp_path):
    first = _first_pool(train_samples, 5)
    target = next(s for s in first if s["label"]["tool_calls"])
    th = tmp_path / "th.jsonl"
    _write_thoughts(th, first)
    rows = [json.loads(line) for line in th.read_text(encoding="utf-8").splitlines()]
    next(r for r in rows if r["id"] == target["id"])["steps"][0]["call"] = "other.tool"
    th.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=target["id"]):
        _run(tmp_path, th, n=5)


def test_thoughts_need_prompt_v3(train_samples, tmp_path):
    _write_thoughts(tmp_path / "th.jsonl", _first_pool(train_samples, 2))
    with pytest.raises(SystemExit):
        _run(tmp_path, tmp_path / "th.jsonl", n=2, prompt="v1")


def test_thoughts_reject_extra_samples(train_samples, tmp_path):
    _write_thoughts(tmp_path / "th.jsonl", _first_pool(train_samples, 2))
    with pytest.raises(SystemExit):
        _run(tmp_path, tmp_path / "th.jsonl", "--extra-samples", str(tmp_path / "x.jsonl"), n=2)


def test_load_thoughts_ignores_a_torn_last_line(tmp_path):
    path = tmp_path / "th.jsonl"
    path.write_text('{"id": "a", "steps": []}\n{"id": "b", "ste', encoding="utf-8")
    assert list(load_thoughts(path)) == ["a"]
