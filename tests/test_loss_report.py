import json

import pytest

from lifemate.loss_report import analyze, failure, gold_tool, load_run, pattern, render

CALL = lambda name, **args: {"name": name, "arguments": args}  # noqa: E731
ASK = CALL("sys.ask_missing_info", tool_name="phone.call", params=["contact_id"])


@pytest.mark.parametrize("text, expected", [
    ("Đặt báo thức sáu giờ, à không, bảy giờ", "self-correction"),
    ("Nếu mai mưa thì huỷ lịch", "condition"),
    ("Ví không đủ thì lấy ngân hàng", "fallback"),
    ("Đặt chuyến rẻ nhất", "pick best"),
    ("Chuyển một nửa tiền điện", "arithmetic"),
    ("Báo thức trước giờ họp một tiếng", "relative time"),
    ("Gọi cho người đó", "refers back"),
    ("Tắt đèn rồi mở nhạc", "several steps"),
    ("Mở nhạc", "plain"),
])
def test_pattern_takes_the_first_match(text, expected):
    assert pattern(text) == expected


def test_failure_separates_state_errors_from_call_errors():
    gold = [CALL("a.b", x=1)]
    assert failure({"score": 1.0, "hash_ok": True}, gold, gold) == "no loss"
    assert failure({"score": 0.5, "hash_ok": True}, gold, gold + [CALL("c.d")]) == "state right, calls differ"
    assert failure({"score": 0.0, "hash_ok": False}, [], gold) == "out of scope, made calls"
    assert failure({"score": 0.0, "hash_ok": False}, gold, []) == "no calls"
    assert failure({"score": 0.0, "hash_ok": False}, gold, [ASK]) == "ask/act flip"
    assert failure({"score": 0.0, "hash_ok": False}, gold, [CALL("c.d", x=1)]) == "other tools"
    assert failure({"score": 0.0, "hash_ok": False}, gold, [CALL("a.b", x=2)]) == "args: number"


def test_gold_tool_names_what_the_request_wants_done():
    assert gold_tool([]) == "(out of scope)"
    assert gold_tool([CALL("contacts.search"), CALL("phone.call")]) == "phone.call"
    assert gold_tool([CALL("contacts.search"), ASK]) == "ask: phone.call"


SAMPLES = {
    "a": {"audio_text": "Nếu mưa thì gọi Minh", "history": [], "label": {"tool_calls": [CALL("phone.call", id="c1")]}},
    "b": {"audio_text": "Mở nhạc", "history": [], "label": {"tool_calls": [CALL("music.play")]}},
    "c": {"audio_text": "Tắt đèn", "history": [], "label": {"tool_calls": [CALL("light.set")]}},
    "d": {"audio_text": "Tắt quạt", "history": [], "label": {"tool_calls": [CALL("fan.set")]}},
}
ROWS = {"a": {"score": 0.0, "hash_ok": False, "kind": "act"}, "b": {"score": 1.0, "hash_ok": True, "kind": "act"},
        "c": {"score": 0.5, "hash_ok": True, "kind": "act"}, "d": {"score": 1.0, "hash_ok": True, "kind": "act"}}
PREDS = {"a": {"tool_calls": [CALL("phone.call", id="c2")]}, "b": {"tool_calls": [CALL("music.play")]},
         "c": {"tool_calls": [CALL("light.set"), CALL("x.y")]}, "d": {"tool_calls": [CALL("fan.set")]}}


def test_every_table_adds_up_to_the_points_lost():
    rows = analyze(ROWS, PREDS, SAMPLES)
    text = render("run", rows, top=5)
    assert "4 samples, score 62.50, 37.50 points lost." in text
    assert "| condition | 1 | 25.00 |" in text
    assert "| state right, calls differ | 1 | 12.50 |" in text
    assert "| condition / args: id | 1 | 25.00 |" in text  # the subset table still divides by the whole run


def _run(tmp_path, asr="gold", split="train"):
    run = tmp_path / "r"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"asr": asr, "split": split}), encoding="utf-8")
    (run / "per_sample.jsonl").write_text("".join(json.dumps({"id": i, **r}) + "\n" for i, r in ROWS.items()), encoding="utf-8")
    (run / "submission.jsonl").write_text("".join(json.dumps({"id": i, **p}) + "\n" for i, p in PREDS.items()), encoding="utf-8")
    return run


def test_load_run_refuses_asr_runs_unless_allowed(tmp_path):
    run = _run(tmp_path, asr="cached")
    with pytest.raises(SystemExit, match="not gold"):
        load_run(run, SAMPLES, allow_asr=False)
    assert len(load_run(run, SAMPLES, allow_asr=True)) == 4


def test_load_run_refuses_splits_without_labels(tmp_path):
    with pytest.raises(SystemExit, match="only works on train"):
        load_run(_run(tmp_path, split="public_test"), SAMPLES, allow_asr=False)
