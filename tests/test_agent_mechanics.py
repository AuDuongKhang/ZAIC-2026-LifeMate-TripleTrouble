from lifemate.agents.fewshot import ExampleBank, render_examples
from lifemate.agents.refs import spoken_numbers, ungrounded_phones


def _doc(sid, text, calls, results=None):
    return {"id": sid, "audio_text": text, "history": [], "context": {"installed_apps": []},
            "label": {"tool_calls": calls, "tool_results": results or [{"ok": True}] * len(calls)}}


def _bank():
    docs = [_doc("e1", "kể chuyện cười đi", []), _doc("e2", "hát một bài đi", []), _doc("e3", "đặt vé tàu đi", []),
            _doc("a1", "đặt báo thức sáu giờ", [{"name": "alarm.create", "arguments": {"time": "06:00"}}])]
    return ExampleBank(docs, {d["id"] for d in docs})


def test_max_empty_caps_out_of_scope_examples_and_needs_a_shared_word():
    bank = _bank()
    plain = bank.select("đặt vé tàu đi Huế", {"alarm"}, k=4)
    assert sum(not d["label"]["tool_calls"] for d in plain) == 3
    capped = bank.select("đặt vé tàu đi Huế", {"alarm"}, k=4, max_empty=1)
    assert [d["id"] for d in capped if not d["label"]["tool_calls"]] == ["e3"]
    assert "a1" in [d["id"] for d in capped]
    assert all(d["label"]["tool_calls"] for d in bank.select("zzz", {"alarm"}, k=4, max_empty=1))


def test_render_examples_cut_is_configurable():
    ex = [_doc("x", "q", [{"name": "a.b", "arguments": {}}], [{"ok": True, "items": ["y" * 50] * 20}])]
    assert "..." in render_examples(ex)
    assert "..." not in render_examples(ex, max_result_chars=5000)


def test_spoken_numbers_reads_digit_words_and_digit_runs():
    assert "09842153762" in spoken_numbers("Nạp tiền cho số không chín tám bốn hai một năm ba bảy sáu hai giúp chị")
    assert "0912345678" in spoken_numbers("gọi số 0912 345 678 nhé")
    assert "0912345678" in spoken_numbers("+84 912345678")


def test_ungrounded_phones_blocks_made_up_numbers_only():
    seen = {"09416258779", "c_317"}
    assert ungrounded_phones({"phone_number": "0912345678"}, seen, ["gọi cho Sarah"]) == ["phone_number='0912345678'"]
    assert ungrounded_phones({"phone_number": "09416258779"}, seen, ["gọi cho Sarah"]) == []
    assert ungrounded_phones({"phone": "09842153762"}, set(),
                             ["số không chín tám bốn hai một năm ba bảy sáu hai"]) == []
    assert ungrounded_phones({"contact_id": "c_317"}, seen, []) == []
