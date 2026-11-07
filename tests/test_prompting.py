from lifemate.agents.prompting import build_messages, history_messages, system_prompt, tool_schema
from lifemate.task import Task

CTX = {"current_time": "2026-11-13T12:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh", "location": {"lat": 10.7, "lng": 106.7},
       "user_profile": {"name": "Anh Khoa", "home_address": "12 Nguyễn Huệ"}, "installed_apps": {"sms": {"tools": ["sms.send"]}}}


def test_tool_schema_one_of_and_fuzzy():
    spec = {"name": "sms.send", "kind": "write", "app": "sms", "returns": "", "description": "Send SMS",
            "parameters": {"type": "object", "properties": {
                "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                "phone_number": {"type": "string"},
                "content": {"type": "string", "x-match": "fuzzy", "description": "Body"}},
                "required": ["content"], "x-one-of": [["contact_id"], ["phone_number"]]}}
    fn = tool_schema(spec)["function"]
    assert fn["name"] == "sms.send" and fn["description"].endswith("Provide one of: contact_id | phone_number.")
    assert "x-one-of" not in fn["parameters"] and "x-ref" not in fn["parameters"]["properties"]["contact_id"]
    assert fn["parameters"]["properties"]["contact_id"]["description"] == "(id from contacts.contacts)"
    assert fn["parameters"]["properties"]["content"]["description"] == "Body (free text)"
    assert spec["parameters"]["properties"]["content"]["description"] == "Body"  # catalog spec left untouched


def test_tool_schema_keeps_nested_items():
    spec = {"name": "sys.ask_missing_info", "description": "Ask", "parameters": {"type": "object", "properties": {
        "params": {"type": "array", "items": {"type": "string"}}}, "required": ["params"]}}
    assert tool_schema(spec)["function"]["parameters"]["properties"]["params"]["items"] == {"type": "string"}


def test_system_prompt_has_context_values():
    text = system_prompt(CTX)
    assert "2026-11-13T12:00:00+07:00" in text and "12 Nguyễn Huệ" in text and "sys.ask_missing_info" in text
    assert "installed_apps" not in text


def test_date_hints_append_the_table():
    from lifemate.agents.dates import date_table
    text = system_prompt(CTX, date_hints=True)
    assert text.startswith(system_prompt(CTX)) and text.endswith("\n\n" + date_table(CTX["current_time"]))
    assert "2026-11-13 thứ sáu (hôm nay, tuần này)" in text
    task = Task(id="t", request="bật đèn", history=[], tools=[], context=CTX)
    msgs = build_messages(task, count_tokens=len, budget=10, examples_text="EXAMPLES", date_hints=True)
    assert msgs[0]["content"] == text + "\n\nEXAMPLES"  # the table follows Context, before the examples


def test_date_hints_off_by_default():
    assert system_prompt(CTX) == system_prompt(CTX, date_hints=False)
    assert "Calendar" not in system_prompt(CTX) and "hôm nay" not in system_prompt(CTX)
    task = Task(id="t", request="bật đèn", history=[], tools=[], context=CTX)
    assert build_messages(task, count_tokens=len, budget=10) == build_messages(
        task, count_tokens=len, budget=10, date_hints=False)


def test_history_messages_pairs_calls_with_results():
    msgs = history_messages([{"role": "user", "text": "báo thức mai?"},
        {"role": "assistant", "text": "Có 1.", "tool_calls": [{"name": "alarm.list", "arguments": {}}], "tool_results": [{"ok": True}]}])
    # The assistant's text was said after it saw the results, so it comes after the tool messages.
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "alarm.list" and msgs[1]["content"] == ""
    assert msgs[2]["content"] == '{"ok": true}' and msgs[3]["content"] == "Có 1."


def test_build_messages_drops_oldest_history_first():
    task = Task(id="t", request="tắt đèn", context=CTX, tools=[], history=[
        {"role": "user", "text": "u1"}, {"role": "assistant", "text": "a1"},
        {"role": "user", "text": "u2"}, {"role": "assistant", "text": "a2"}])
    msgs = build_messages(task, count_tokens=len, budget=4)   # 1 token per message
    assert [m.get("content") for m in msgs[1:]] == ["u2", "a2", "tắt đèn"]
    assert msgs[0]["role"] == "system"


def test_build_messages_never_drops_request():
    task = Task(id="t", request="tắt đèn", context=CTX, tools=[], history=[{"role": "user", "text": "u1"}])
    msgs = build_messages(task, count_tokens=len, budget=1)
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[-1]["content"] == "tắt đèn"


def test_prompt_versions_are_kept():
    from lifemate.agents.prompts import DEFAULT_PROMPT, PROMPTS
    assert PROMPTS["v1"].startswith("You are a voice assistant") and "Never invent" not in PROMPTS["v1"]
    assert "Never invent" in PROMPTS["v2"] and DEFAULT_PROMPT == "v1"  # default = best measured version
    assert PROMPTS["v2"] in system_prompt(CTX, "v2") and PROMPTS["v1"] in system_prompt(CTX)


def test_unknown_prompt_version_fails_loudly():
    import pytest
    with pytest.raises(KeyError):
        system_prompt(CTX, "v999")


def test_build_messages_appends_examples_to_the_system_prompt_only():
    task = Task(id="t", request="bật đèn", history=[], tools=[], context=CTX)
    plain = build_messages(task, count_tokens=len, budget=10)
    with_examples = build_messages(task, count_tokens=len, budget=10, examples_text="EXAMPLES")
    assert with_examples[0]["content"] == plain[0]["content"] + "\n\nEXAMPLES"
    assert with_examples[1:] == plain[1:]
    assert build_messages(task, count_tokens=len, budget=10, examples_text="") == plain


def test_v3_is_v1_plus_one_rule():
    from lifemate.agents.prompts import DEFAULT_PROMPT, PROMPTS
    extra = PROMPTS["v3"][len(PROMPTS["v1"]):]
    assert PROMPTS["v3"].startswith(PROMPTS["v1"]) and extra.startswith("\n7. ") and extra.count("\n") == 1
    assert "Vietnamese" in extra and DEFAULT_PROMPT == "v1"
