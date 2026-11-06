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


CATALOG = {
    "news.search": {"kind": "read", "parameters": {"properties": {"query": {}}}},
    "news.headlines": {"kind": "read", "parameters": {"properties": {"topic": {}}}},
    "notes.create": {"kind": "write", "parameters": {"properties": {"title": {"x-match": "fuzzy"}, "content": {"x-match": "fuzzy"}}}},
    "contacts.search": {"kind": "read", "parameters": {"properties": {"query": {}}}},
    "phone.call": {"kind": "write", "parameters": {"properties": {"contact_id": {}}}},
}


def _issues(gold, pred):
    from lifemate.data import fuzzy_params
    from lifemate.loss_report import state_right_issues
    return state_right_issues(gold, pred, CATALOG, fuzzy_params(CATALOG))


def test_issue_for_a_read_query_that_is_not_identical():
    assert _issues([CALL("news.search", query="xăng")], [CALL("news.search", query="giá xăng")]) == ["read args: value"]


def test_issue_for_the_wrong_read_tool_of_the_same_app():
    assert _issues([CALL("news.search", query="khoa học")], [CALL("news.headlines", topic="tech")]) == ["other read tool, same app"]


def test_issue_for_free_text_names_its_shape():
    gold = CALL("notes.create", title="Tin cần đọc", content="Vườn rau dùng tưới nhỏ giọt")
    long = CALL("notes.create", title="Tin cần đọc", content="Vườn rau dùng tưới nhỏ giọt, tiết kiệm nước gần một nửa so với tưới tràn trên toàn bộ khu vườn")
    other = CALL("notes.create", title="Tin cần đọc", content="Bài báo nói chuyện khác hẳn luôn")
    assert _issues([gold], [long]) == ["free text: longer than gold"]
    assert _issues([gold], [other]) == ["free text: other words"]


def test_issues_for_missing_and_extra_calls_keep_their_kind():
    assert _issues([ASK], []) == ["missing ask"]
    assert _issues([], [CALL("contacts.search", query="Lan")]) == ["extra read"]
    assert _issues([CALL("contacts.search", query="Lan")], []) == ["missing read"]


def test_exact_matches_leave_no_issue_and_are_not_paired_with_others():
    a, b = CALL("contacts.search", query="Lan"), CALL("contacts.search", query="Minh")
    assert _issues([a, b], [b, a]) == []
    assert _issues([a], [b, a]) == ["extra read"]


def test_issue_table_splits_a_sample_equally_and_adds_up():
    from lifemate.loss_report import issue_table
    rows = [{"id": "a", "loss": 100.0, "issues": ["extra read", "missing read"]}, {"id": "b", "loss": 0.0, "issues": []}]
    text = "\n".join(issue_table(rows))
    assert "| extra read | 1 | 25.00 |" in text and "| missing read | 1 | 25.00 |" in text
