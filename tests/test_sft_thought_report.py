"""Reports for the stop points: sentences without a call in a run, and a sheet for reading sentences by hand."""
import json

import pytest

from lifemate.data import default_data_root, import_zaic_tools
from lifemate.sft import thought_report
from lifemate.sft.thought_report import read_sheet, run_report, sentence_without_call


def _row(stop, *outputs, seconds=1.0):
    return {"id": "s", "request": "r", "calls": [], "seconds": seconds, "error": None,
            "agent": {"stop": stop, "steps": [{"output": o, "rejected": []} for o in outputs]}}


@pytest.mark.parametrize("row, expected", [
    (_row("no_call", "Cần tìm Minh trước."), True),
    (_row("no_call", "<tool_call>{}</tool_call>", "Giờ gọi cho Minh."), True),
    (_row("no_call", "Đã xong."), False),
    (_row("no_call", "Người dùng tự huỷ.\nXin lỗi, tôi không làm được việc này."), False),
    (_row("no_call", ""), False),
    (_row("asked", "Thiếu số tiền.\n<tool_call>{}</tool_call>"), False),
    (_row("time"), False),
    ({"id": "s", "agent": None, "seconds": 1.0, "error": "x"}, False),
])
def test_sentence_without_call(row, expected):
    assert sentence_without_call(row) is expected


def test_run_report_counts_share_and_seconds():
    report = run_report([_row("no_call", "Câu.", seconds=2.0), _row("asked", "x", seconds=31.0)])
    assert report == {"n": 2, "no_call_sentences": 1, "share": 50.0, "mean_seconds": 16.5, "max_seconds": 31.0, "over_30s": 1}


def _write_run(tmp_path, rows):
    run = tmp_path / "run"
    run.mkdir()
    (run / "trace.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return run


def test_run_form_prints_the_report_and_flags_a_share_over_two_percent(tmp_path, capsys):
    run = _write_run(tmp_path, [_row("no_call", "Câu."), _row("asked", "x")])
    thought_report.main([str(run), "--show", "1"])
    out = capsys.readouterr().out
    assert '"no_call_sentences": 1' in out and "over the 2% limit" in out and "request: r" in out


def test_run_form_stays_quiet_about_the_limit_when_the_share_is_small(tmp_path, capsys):
    run = _write_run(tmp_path, [_row("asked", "x")] * 3)
    thought_report.main([str(run)])
    assert "over the 2% limit" not in capsys.readouterr().out


def test_read_sheet_shows_what_a_reader_needs(train_samples, tmp_path):
    root = default_data_root()
    zaic_tools = import_zaic_tools(root, "train")
    picked = [s for s in train_samples if len(s["label"]["tool_calls"]) >= 3 and not s["history"]
              and all(c["name"] != "sys.ask_missing_info" for c in s["label"]["tool_calls"])][:2]
    rows = [{"id": s["id"], "teacher": "t", "revision": "r",
             "steps": [{"call": c["name"], "thought": f"Câu {i}."} for i, c in enumerate(s["label"]["tool_calls"])]}
            for s in picked]
    by_id = {s["id"]: s for s in train_samples}
    sheet = read_sheet(rows, by_id, zaic_tools, root, 2)
    assert sheet == read_sheet(rows, by_id, zaic_tools, root, 2)  # the seed fixes the pick
    for s in picked:
        assert s["audio_text"] in sheet
    for s, row in zip(picked, rows):
        section = sheet.split(f"## ")[[x.split()[1] for x in sheet.split("## ")[1:]].index(s["id"]) + 1]
        k = next(i for i in range(len(row["steps"])) if f"Câu {i}." in section)
        assert json.dumps(s["label"]["tool_calls"][k]["arguments"], ensure_ascii=False) in section
        env = zaic_tools.ToolEnv("train", s["id"], root=root)
        for call in s["label"]["tool_calls"][:k]:
            result = env.call(call["name"], call["arguments"])
            assert json.dumps(result, ensure_ascii=False) in section  # the earlier results the sentence may rely on


def test_read_sheet_takes_n_samples_and_one_step_each(train_samples):
    root = default_data_root()
    zaic_tools = import_zaic_tools(root, "train")
    picked = [s for s in train_samples if s["label"]["tool_calls"]][:6]
    rows = [{"id": s["id"], "teacher": "t", "revision": "r",
             "steps": [{"call": c["name"], "thought": f"Câu {i}."} for i, c in enumerate(s["label"]["tool_calls"])]}
            for s in picked]
    sheet = read_sheet(rows, {s["id"]: s for s in train_samples}, zaic_tools, root, 4)
    assert sheet.count("\n## ") + sheet.startswith("## ") == 4


def test_read_form_writes_the_sheet_next_to_the_thoughts_file(train_samples, tmp_path):
    picked = [s for s in train_samples if s["label"]["tool_calls"]][:3]
    path = tmp_path / "pilot.jsonl"
    path.write_text("".join(json.dumps({"id": s["id"], "teacher": "t", "revision": "r", "steps": [
        {"call": c["name"], "thought": "Câu."} for c in s["label"]["tool_calls"]]}, ensure_ascii=False) + "\n" for s in picked),
        encoding="utf-8")
    thought_report.main(["--thoughts", str(path), "--read", "2"])
    sheet = (tmp_path / "pilot.read.md").read_text(encoding="utf-8")
    assert sheet.count("Câu.") == 2


def test_one_form_at_a_time(tmp_path):
    with pytest.raises(SystemExit):
        thought_report.main([])
    with pytest.raises(SystemExit):
        thought_report.main([str(tmp_path), "--thoughts", str(tmp_path / "x.jsonl"), "--read", "5"])
