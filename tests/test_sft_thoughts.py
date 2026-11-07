"""Teacher prompt, sentence checks and the per-sample writer of lifemate.sft.thoughts (inline samples, a fake LLM)."""
import copy
import json
from collections import Counter

import pytest

from lifemate.sft.thoughts import banned_words, check_sentence, step_calls, step_prompt, write_sample

CONTEXT = {"current_time": "2026-01-01T08:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh", "location": {"lat": 10.7, "lng": 106.7},
           "user_profile": {"name": "Anh Khoa"},
           "installed_apps": {"contacts": {"tools": ["contacts.search"]}, "phone": {"tools": ["phone.call"]}}}
HISTORY = [{"role": "user", "audio_text": "lời chào trước đó"}, {"role": "assistant", "text": "Chào anh."}]
SAMPLE = {"id": "s1", "audio_text": "gọi cho Minh", "history": HISTORY, "context": CONTEXT, "label": {"tool_calls": [
    {"name": "contacts.search", "arguments": {"query": "Minh"}},
    {"name": "phone.call", "arguments": {"contact_id": "c1"}}]}}
ASK_SAMPLE = {"id": "s2", "audio_text": "gọi cho anh ấy", "history": [], "context": CONTEXT, "label": {"tool_calls": [
    {"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}]}}
EMPTY_SAMPLE = {"id": "s3", "audio_text": "thôi khỏi", "history": [], "context": CONTEXT, "label": {"tool_calls": []}}


def _spec(name, description, params=None):
    props = {p: {"type": "string"} for p in params or []}
    return {"name": name, "kind": "read", "app": name.split(".")[0], "returns": "", "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}


CATALOG = {s["name"]: s for s in [
    _spec("contacts.search", "Search contacts", ["query"]),
    _spec("phone.call", "Call a contact", ["contact_id"]),
    _spec("sys.ask_missing_info", "Ask the user for missing parameters", ["tool_name", "params"]),
    _spec("weather.today", "Weather today")]}


def _catalog_with(name):
    catalog = copy.deepcopy(CATALOG)
    catalog["contacts.search"]["parameters"]["properties"][name] = {"type": "string"}
    return catalog


def _text(messages):
    return "\n".join(m["content"] for m in messages)


class FakeEnv:
    def __init__(self):
        self.calls = 0

    def call(self, name, arguments):
        n = self.calls
        self.calls += 1
        return {"items": [{"id": "c1", "name": f"RESULT{n}"}]}


class FakeLLM:
    def __init__(self, texts):
        self.texts, self.seen = list(texts), []

    def generate(self, messages):
        self.seen.append(copy.deepcopy(messages))
        return self.texts.pop(0)


def test_out_of_scope_has_one_step():
    assert step_calls({"label": {"tool_calls": []}}) == [None]


def test_prompt_shows_the_past_and_the_next_action_only():
    text = _text(step_prompt(SAMPLE, CATALOG, 1, [{"items": [{"name": "RESULT0"}]}]))
    assert "RESULT0" in text and "phone.call" in text and "c1" in text  # earlier result, next action and its argument
    assert "RESULT1" not in text


def test_first_prompt_hides_later_calls_and_their_tools():
    text = _text(step_prompt(SAMPLE, CATALOG, 0, []))
    assert "phone.call" not in text and "Call a contact" not in text


def test_prompt_shows_context_history_request_and_used_tools_only():
    text = _text(step_prompt(SAMPLE, CATALOG, 0, []))
    for part in ("gọi cho Minh", "lời chào trước đó", "2026-01-01T08:00:00+07:00", "Search contacts"):
        assert part in text
    assert "Weather today" not in text


def test_an_ask_prompt_describes_the_tool_asked_about():
    assert "Call a contact" in _text(step_prompt(ASK_SAMPLE, CATALOG, 0, []))


def test_an_out_of_scope_prompt_names_the_installed_apps_and_says_nothing_is_called():
    assert "contacts" in _text(step_prompt(EMPTY_SAMPLE, CATALOG, 0, []))


@pytest.mark.parametrize("text, reason", [
    ("", "empty"),
    (" ".join(["từ"] * 31), "too long"),
    ("Câu một. Câu hai.", "several sentences"),
    ("Số tiền là 1.07 triệu.", None),
    ("Dòng một\nDòng hai", "several sentences"),
    ("Đã xong.", "closing text"),
    ("Dùng <tool_call> ngay.", "markup"),
    ("Cần gọi contact_id trước.", "names contact_id"),
    ("Cần tìm người tên Minh.", None),
])
def test_check_sentence(text, reason):
    assert check_sentence(text, {"contact_id", "phone"}) == reason


def test_banned_words_cover_used_tools_and_the_tool_asked_about():
    assert {"contacts", "search", "contacts.search", "query", "phone", "contact_id"} <= banned_words(SAMPLE, CATALOG, [])
    assert {"tool_name", "phone", "contact_id"} <= banned_words(ASK_SAMPLE, CATALOG, [])


def test_a_word_the_user_said_or_a_result_value_is_allowed_but_a_result_key_is_not():
    results = [{"query": 1, "label": "phone"}]
    banned = banned_words(SAMPLE, CATALOG, results)
    assert "phone" not in banned and "query" in banned
    assert "minh" not in banned_words(SAMPLE, _catalog_with("minh"), [])
    assert "sodu" in banned_words(SAMPLE, _catalog_with("sodu"), [])


def test_names_shorter_than_three_letters_are_not_banned():
    assert "to" not in banned_words(SAMPLE, _catalog_with("to"), [])


def test_write_sample_runs_each_call_after_its_sentence():
    env, llm = FakeEnv(), FakeLLM(["Cần tìm Minh trước.", "Đã thấy Minh, gọi luôn."])
    row, failure = write_sample(SAMPLE, CATALOG, env, llm, "t", "r", Counter())
    assert failure is None and env.calls == 2
    assert row == {"id": "s1", "teacher": "t", "revision": "r", "steps": [
        {"call": "contacts.search", "thought": "Cần tìm Minh trước."},
        {"call": "phone.call", "thought": "Đã thấy Minh, gọi luôn."}]}
    assert "RESULT0" not in _text(llm.seen[0]) and "RESULT0" in _text(llm.seen[1])


def test_a_failed_sentence_is_retried_once_with_the_failure_named():
    llm = FakeLLM(["Cần contact_id.", "Cần tìm Minh trước.", "Gọi luôn."])
    stats = Counter()
    row, _ = write_sample(SAMPLE, CATALOG, FakeEnv(), llm, "t", "r", stats)
    assert row is not None and "names contact_id" in _text(llm.seen[1])
    assert stats["first_try"] == 1 and stats["retried"] == 1


def test_two_failures_leave_the_sample_out_and_stop_its_later_steps():
    llm = FakeLLM(["", "", "never used"])
    stats = Counter()
    row, failure = write_sample(SAMPLE, CATALOG, FakeEnv(), llm, "t", "r", stats)
    assert row is None and failure["id"] == "s1" and failure["step"] == 0 and failure["reason"] == "empty"
    assert len(llm.seen) == 2 and stats["failed"] == 1 and stats["reason: empty"] == 1


def test_out_of_scope_writes_one_step_and_calls_nothing():
    env = FakeEnv()
    row, _ = write_sample(EMPTY_SAMPLE, CATALOG, env, FakeLLM(["Người dùng tự huỷ yêu cầu."]), "t", "r", Counter())
    assert row["steps"] == [{"call": None, "thought": "Người dùng tự huỷ yêu cầu."}] and env.calls == 0
