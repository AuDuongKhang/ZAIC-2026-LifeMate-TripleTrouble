import json

from lifemate.asr_impact import analyze, call_change, error_categories, main, render, value_kind

SAMPLES = {
    "a": {"audio_text": "Gọi cho Minh nhé", "history": []},
    "b": {"audio_text": "Bật đèn phòng khách", "history": []},
    "c": {"audio_text": "Đặt báo thức sáu giờ", "history": []},
}
GOLD = {i: {"id": i, "kind": "act", "score": 1.0} for i in SAMPLES}
ASR = {"a": {"id": "a", "kind": "act", "score": 0.0}, "b": {"id": "b", "kind": "act", "score": 1.0},
       "c": {"id": "c", "kind": "act", "score": 0.5}}
CALL = lambda name, **args: {"name": name, "arguments": args}  # noqa: E731
GP = {"a": {"tool_calls": [CALL("phone.call", contact_id="c1")]}, "b": {"tool_calls": [CALL("smart_home.light_set", power="on")]},
      "c": {"tool_calls": [CALL("alarm.create", time="06:00")]}}
AP = {"a": {"tool_calls": [CALL("phone.call", contact_id="c2")]}, "b": {"tool_calls": [CALL("smart_home.light_set", power="on")]},
      "c": {"tool_calls": [CALL("alarm.create", time="05:00")]}}
REQ = {"a": "Gọi cho Mình nhé", "b": "Bật đèn phòng khách", "c": "Đặt báo thức bảy giờ"}


def test_error_categories_and_value_kinds():
    assert error_categories("Gọi cho Minh nhé", "Gọi cho Mình nhé") == {"name"}
    assert error_categories("Đặt báo thức sáu giờ", "Đặt báo thức sáu giờ") == set()
    assert value_kind("contact_id", "c1") == "id" and value_kind("time", "2026-11-20T09:00:00+07:00") == "date/time"
    assert value_kind("quantity", 3) == "number" and value_kind("title", "đi chợ") == "text/other"


def test_call_change_names_the_difference():
    ask = CALL("sys.ask_missing_info", tool_name="phone.call", params=["contact_id"])
    assert call_change([CALL("a.b", x=1)], [CALL("a.b", x=1)]) == "same calls"
    assert call_change([CALL("phone.call", contact_id="c1")], [ask]) == "ask/act flip"
    assert call_change([CALL("a.b")], [CALL("c.d")]) == "other tools"
    assert call_change([CALL("a.b", contact_id="1", n=2)], [CALL("a.b", contact_id="2", n=3)]) == "args: id+number"


def test_analyze_assigns_cause_and_loss():
    rows = {r["id"]: r for r in analyze(SAMPLES, GOLD, ASR, GP, AP, REQ)}
    assert rows["a"]["loss"] == 100 and rows["a"]["cause"] == "final:name" and rows["a"]["change"] == "args: id"
    assert rows["b"]["loss"] == 0 and rows["b"]["cause"] == "exact" and rows["b"]["change"] == "same calls"
    assert rows["c"]["loss"] == 50 and rows["c"]["cause"] == "final:number"
    history = {r["id"]: r for r in analyze(SAMPLES, GOLD, ASR, GP, AP, REQ, {"b": 0.2})}
    assert history["b"]["cause"] == "history"


def test_render_reports_totals():
    text = render(analyze(SAMPLES, GOLD, ASR, GP, AP, REQ), top=2)
    assert "Mean score loss +50.00" in text and "final:name" in text and "`a` loss 100" in text


def test_main_writes_the_report(tmp_path, train_samples):
    a = train_samples[0]
    for name, score in (("gold", 1.0), ("asr", 0.0)):
        run = tmp_path / name
        run.mkdir()
        (run / "per_sample.jsonl").write_text(json.dumps({"id": a["id"], "kind": "act", "score": score}) + "\n")
        (run / "submission.jsonl").write_text(json.dumps({"id": a["id"], "tool_calls": a["label"]["tool_calls"]}) + "\n")
    (tmp_path / "asr" / "trace.jsonl").write_text(json.dumps({"id": a["id"], "request": "xin chào"}) + "\n")
    rows = main([str(tmp_path / "gold"), str(tmp_path / "asr")])
    assert rows[0]["loss"] == 100 and (tmp_path / "asr" / "asr_impact.md").exists()
