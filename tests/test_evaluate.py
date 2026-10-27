import json

from lifemate.evaluate import summary_row


def test_summary_row_notes_prompt_version(tmp_path):
    meta = {"finished": "2026-10-27T07:00:00", "git": {"commit": "abc1234"}, "split": "train", "n": 100, "agent": "react",
            "asr": "gold", "mean_seconds": 3.0, "max_seconds": 8.0, "agent_info": {"prompt": "v2", "check_refs": True}}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    row = summary_row(tmp_path)
    assert row.endswith("|") and "prompt v2" in row
