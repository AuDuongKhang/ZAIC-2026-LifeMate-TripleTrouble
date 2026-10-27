from lifemate.agents.parsing import parse_tool_calls


def test_two_calls():
    t = ('<tool_call>\n{"name": "alarm.list", "arguments": {}}\n</tool_call>\n'
         '<tool_call>{"name": "a.b", "arguments": "{\\"x\\": 1}"}</tool_call>')
    assert parse_tool_calls(t) == ([{"name": "alarm.list", "arguments": {}}, {"name": "a.b", "arguments": {"x": 1}}], [])


def test_plain_text_means_no_call():
    assert parse_tool_calls("Đã tắt đèn.") == ([], [])


def test_broken_json_is_an_error_not_a_call():
    calls, errors = parse_tool_calls('<tool_call>{"name": "alarm.list", "arguments": {</tool_call>')
    assert calls == [] and len(errors) == 1


def test_missing_name_is_an_error():
    calls, errors = parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>')
    assert calls == [] and len(errors) == 1


def test_missing_arguments_means_empty():
    assert parse_tool_calls('<tool_call>{"name": "alarm.list"}</tool_call>') == ([{"name": "alarm.list", "arguments": {}}], [])


def test_unclosed_block_at_end_of_generation():
    calls, errors = parse_tool_calls('<tool_call>{"name": "alarm.list", "arguments": {}}')
    assert calls == [{"name": "alarm.list", "arguments": {}}] and errors == []
