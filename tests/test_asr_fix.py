"""LLM transcript fix: prompts hold no gold text, bad answers are dropped, the run resumes and writes a cache."""
import io
import json

from lifemate.asr_fix import build_messages, clean_answer, fix_samples, report

SAMPLE = {"id": "s1", "audio": "a/s1.wav", "audio_text": "GOLD đặt báo thức", "label": {"tool_calls": []},
          "context": {"installed_apps": {"alarm": {}, "zzz": {}}, "user_profile": {"name": "Bác Tư"}},
          "history": [{"role": "user", "audio": "a/s1_u1.wav", "audio_text": "GOLD câu trước"},
                      {"role": "assistant", "text": "Dạ vâng."}]}
ROWS = {"a/s1.wav": {"audio": "a/s1.wav", "text": "đặt báo thứ", "seconds": 0.1},
        "a/s1_u1.wav": {"audio": "a/s1_u1.wav", "text": "câu trước nghe được", "seconds": 0.1}}
APPS = {"alarm": "Báo thức"}


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def generate(self, messages, tools, max_time=None):
        self.calls += 1
        return self.answer


def test_prompt_uses_asr_text_and_no_gold():
    text = build_messages(SAMPLE, "đặt báo thứ", ROWS, APPS)[1]["content"]
    assert "GOLD" not in text and "câu trước nghe được" in text and "Dạ vâng." in text
    assert "Báo thức" in text and "zzz" in text and "Bác Tư" in text and text.endswith("Transcript: đặt báo thứ")


def test_clean_answer_drops_bad_answers():
    t = "đặt báo thức sáu giờ sáng mai nha"
    assert clean_answer(t, t, 0.3) == (t, "same")
    assert clean_answer(t, "Output: đặt báo thức sáu giờ sáng mai nhé", 0.3)[1] == "changed"
    assert clean_answer(t, "", 0.3) == (t, "dropped: empty")
    assert clean_answer(t, '<tool_call>{"name": "a.b"}</tool_call>', 0.3) == (t, "dropped: markup or several lines")
    assert clean_answer(t, "dòng một\ndòng hai", 0.3) == (t, "dropped: markup or several lines")
    assert clean_answer(t, "Đã xong.", 0.3) == (t, "dropped: too different")


def test_fix_samples_writes_every_clip_and_resumes():
    out = io.StringIO()
    llm = FakeLLM("đặt báo thức")
    result = fix_samples([SAMPLE], ROWS, {}, llm, APPS, 0.5, out)
    assert llm.calls == 1 and result["a/s1.wav"]["text"] == "đặt báo thức" and result["a/s1.wav"]["outcome"] == "changed"
    assert result["a/s1.wav"]["original"] == "đặt báo thứ"
    assert result["a/s1_u1.wav"]["text"] == "câu trước nghe được" and result["a/s1_u1.wav"]["outcome"] == "history"
    assert len(out.getvalue().splitlines()) == 2 and json.loads(out.getvalue().splitlines()[0])["audio"] == "a/s1.wav"
    again = FakeLLM("khác")
    assert fix_samples([SAMPLE], ROWS, result, again, APPS, 0.5, io.StringIO()) == result and again.calls == 0


def test_report_counts_better_and_worse():
    samples = [{**SAMPLE, "id": "s1", "audio": "x1", "audio_text": "đặt báo thức"},
               {**SAMPLE, "id": "s2", "audio": "x2", "audio_text": "bật đèn"}]
    rows = {"x1": {"text": "đặt báo thức", "original": "đặt báo thứ", "outcome": "changed", "seconds": 1.0},
            "x2": {"text": "bật điện", "original": "bật đèn", "outcome": "changed", "seconds": 2.0}}
    text = report("t", samples, rows, {"dev": {"s1", "s2"}})
    assert "| dev | 2 | 0.1667 | 0.2500 | 1 | 1 | 0 |" in text
    assert "`s1`" in text.split("## Improved")[1] and "`s2`" in text.split("## Improved")[0]
