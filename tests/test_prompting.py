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


def test_history_messages_pairs_calls_with_results():
    msgs = history_messages([{"role": "user", "text": "báo thức mai?"},
        {"role": "assistant", "text": "Có 1.", "tool_calls": [{"name": "alarm.list", "arguments": {}}], "tool_results": [{"ok": True}]}])
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "alarm.list"
    assert msgs[2]["content"] == '{"ok": true}'


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
