import os
import subprocess
import sys

import pytest

from lifemate.agents.fewshot import ExampleBank, gold_apps, render_examples, tool_tokens
from lifemate.data import default_data_root
from lifemate.splits import load_splits


def test_select_never_returns_the_sample_itself(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    s = train_samples[0]
    picked = bank.select(s["audio_text"], set(s["context"]["installed_apps"]), k=3, exclude_id=s["id"])
    assert s["id"] not in [p["id"] for p in picked] and len(picked) == 3


def test_select_only_uses_pool(train_samples):
    pool = {s["id"] for s in train_samples[100:]}
    bank = ExampleBank(train_samples, pool_ids=pool)
    picked = bank.select("chuyển tiền cho mẹ", set(train_samples[0]["context"]["installed_apps"]), k=4, exclude_id=None)
    assert picked and all(p["id"] in pool for p in picked)


def test_select_requires_example_apps_installed(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    picked = bank.select("bật đèn phòng khách", {"smart_home"}, k=2, exclude_id=None)
    assert len(picked) == 2 and all(gold_apps(p["label"]) <= {"smart_home"} for p in picked)


def test_select_ranks_the_most_similar_request_first(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    s = next(x for x in train_samples if x["label"]["tool_calls"])
    top = bank.select(s["audio_text"], set(s["context"]["installed_apps"]), k=1, exclude_id=None)[0]
    assert top["audio_text"] == s["audio_text"]


def test_select_offset_returns_the_next_examples(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    s = train_samples[0]
    args = (s["audio_text"], set(s["context"]["installed_apps"]))
    top4 = bank.select(*args, k=4, exclude_id=s["id"])
    assert len(top4) == 4 and bank.select(*args, k=2, exclude_id=s["id"], offset=2) == top4[2:4]
    assert bank.select(*args, k=2, exclude_id=s["id"], offset=0) == top4[:2]


def test_render_examples_shows_calls_and_truncated_results():
    ex = [{"id": "x", "audio_text": "tắt đèn",
           "label": {"tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "l1", "power": "off"}}],
                     "tool_results": [{"ok": True, "pad": "z" * 1000}]}}]
    text = render_examples(ex, max_result_chars=50)
    assert "User: tắt đèn" in text and '<tool_call>{"name": "smart_home.light_set"' in text and "z" * 60 not in text


def test_render_examples_marks_out_of_scope_and_empty_selection():
    out_of_scope = [{"id": "y", "audio_text": "kể chuyện cười", "label": {"tool_calls": [], "tool_results": []}}]
    assert "(no tool call)" in render_examples(out_of_scope)
    assert render_examples([]) == ""


def test_from_data_pool_depends_on_the_split(train_samples):
    from lifemate.agents.fewshot import _sees_its_ids
    from lifemate.data import load_catalog
    root = default_data_root()
    pool, everything = set(load_splits()["pool"]), {s["id"] for s in train_samples}
    on_train, on_public = ExampleBank.from_data(root, "train"), ExampleBank.from_data(root, "public_test")
    assert {d["id"] for d in on_train.docs} <= pool
    visible = {s["id"] for s in train_samples if _sees_its_ids(s, load_catalog(root, "train"))}
    assert {d["id"] for d in on_train.docs} == pool & visible
    assert {d["id"] for d in on_public.docs} == everything & visible


def _sample(sid, text, calls, results=None, history=None):
    return {"id": sid, "audio_text": text, "history": history or [],
            "label": {"tool_calls": calls, "tool_results": results or [{"ok": True}] * len(calls)}}


def test_asked_app_counts_as_a_gold_app():
    ask = _sample("a", "gọi cho Minh", [{"name": "sys.ask_missing_info",
                                         "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}])
    bank = ExampleBank([ask], {"a"})
    assert bank.select("gọi cho Minh", {"weather"}, k=1) == []
    assert bank.select("gọi cho Minh", {"phone"}, k=1) == [ask]


CATALOG = {"sms.send": {"parameters": {"properties": {"contact_id": {"type": "string", "x-ref": "contacts.contacts"}}}},
           "contacts.search": {"parameters": {"properties": {"query": {"type": "string"}}}}}


def test_examples_with_ids_that_never_appear_in_their_own_results_are_skipped():
    blind = _sample("blind", "nhắn cho người đó", [{"name": "sms.send", "arguments": {"contact_id": "c_9"}}])
    seen = _sample("seen", "nhắn cho Minh", [{"name": "contacts.search", "arguments": {"query": "Minh"}},
                                             {"name": "sms.send", "arguments": {"contact_id": "c_9"}}],
                   [{"ok": True, "contacts": [{"contact_id": "c_9"}]}, {"ok": True}])
    bank = ExampleBank([blind, seen], {"blind", "seen"}, catalog=CATALOG)
    assert [e["id"] for e in bank.select("nhắn cho", {"sms", "contacts"}, k=5)] == ["seen"]
    assert len(ExampleBank([blind, seen], {"blind", "seen"})) == 2  # no catalog, no filter


def test_from_data_drops_only_blind_examples(train_samples):
    bank = ExampleBank.from_data(default_data_root(), "train")
    assert 0 < len(bank) < len(load_splits()["pool"])


# --- selection by tool structure ---------------------------------------------------------------------------------------

def spec(name, params, kind="write", description=""):
    return {"name": name, "kind": kind, "description": description,
            "parameters": {"properties": {p: {"type": "string"} for p in params}}}


ANALOG_CATALOG = {s["name"]: s for s in (
    spec("ride.end_ride", ["ride_id"]), spec("calendar.cancel_event", ["event_id"]),
    spec("weather.today", ["city"], kind="read"))}
END_RIDE = _sample("end_ride", "dừng chuyến taxi giúp tôi", [{"name": "ride.end_ride", "arguments": {"ride_id": "r1"}}])
CANCEL = _sample("cancel", "kết thúc phiên họp giúp tôi", [{"name": "calendar.cancel_event",
                                                         "arguments": {"event_id": "e1"}}])
SESSION = spec("parking.end_session", ["session_id"], description="Kết thúc phiên đỗ xe")
ANALOG_REQUEST = "kết thúc phiên đỗ xe"
ANALOG_INSTALLED = {"ride", "calendar", "parking"}


def analog_bank():
    return ExampleBank([END_RIDE, CANCEL], {"end_ride", "cancel"}, catalog=ANALOG_CATALOG)


def test_tool_tokens_are_name_suffix_parameters_and_kind():
    assert tool_tokens(spec("parking.end_session", ["session_id", "plate"])) == {"end", "session", "id", "plate", "write"}
    assert tool_tokens({"name": "x.y"}) == {"y"}


def test_analog_weight_zero_is_bm25():
    bank = analog_bank()
    plain = bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2)
    assert plain[0]["id"] == "cancel"  # the words of the request match this one
    assert bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2, tools=[SESSION], analog_weight=0.0) == plain
    assert bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2, tools=None, analog_weight=5.0) == plain


def test_analog_prefers_same_shape_tool():
    bank = analog_bank()
    picked = bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2, tools=[SESSION], analog_weight=10.0)
    assert [e["id"] for e in picked] == ["end_ride", "cancel"]  # end_ride shares end, id, write with end_session


def test_seen_app_sample_is_unchanged():
    bank = analog_bank()
    seen_tool = spec("ride.end_ride", ["ride_id"], description="Kết thúc phiên đỗ xe")
    plain = bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2)
    for weight in (1.0, 4.0, 100.0):
        assert bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2, tools=[seen_tool], analog_weight=weight) == plain


def test_analog_needs_a_tool_that_shares_words_with_the_request():
    bank = analog_bank()
    unrelated = spec("parking.end_session", ["session_id"], description="Gia hạn thẻ thư viện")
    plain = bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2)
    assert bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2, tools=[unrelated], analog_weight=4.0) == plain


def test_analog_weight_without_a_catalog_is_an_error():
    bank = ExampleBank([END_RIDE], {"end_ride"})
    with pytest.raises(ValueError):
        bank.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=1, tools=[SESSION], analog_weight=1.0)


def test_analog_is_deterministic():
    code = ("from tests.test_fewshot import *; b = analog_bank(); "
            "print([e['id'] for e in b.select(ANALOG_REQUEST, ANALOG_INSTALLED, k=2, tools=[SESSION], analog_weight=10.0)])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": seed,
            "PYTHONUTF8": "1", "PYTHONPATH": os.pathsep.join(["src", "."])}).stdout for seed in ("1", "2", "3")}
    assert len(outs) == 1 and "end_ride" in outs.pop()


def test_report_counts_a_same_shape_tool_as_a_hit_only_with_the_weight():
    from lifemate.fewshot_report import analogous, report
    catalog = {**ANALOG_CATALOG, "parking.end_session": SESSION}
    sample = {**_sample("p", ANALOG_REQUEST, [{"name": "parking.end_session", "arguments": {"session_id": "s1"}}]),
              "context": {"installed_apps": {a: {} for a in ANALOG_INSTALLED}},
              "audio_text": ANALOG_REQUEST}
    sample["context"]["installed_apps"] = {a: {"tools": [t for t in catalog if t.startswith(a + ".")]} for a in ANALOG_INSTALLED}
    bank = analog_bank()
    assert report([sample], bank, catalog, 0.0, k=1) == {"n": 1, "hits": 0, "share": 0.0, "unseen_n": 1, "unseen_hits": 0,
                                                         "unseen_share": 0.0}
    with_weight = report([sample], bank, catalog, 10.0, k=1)
    assert with_weight["hits"] == 1 and with_weight["unseen_share"] == 100.0
    assert analogous(catalog["ride.end_ride"], catalog["parking.end_session"])
    assert not analogous(catalog["calendar.cancel_event"], catalog["weather.today"])


def test_select_leaves_out_excluded_apps(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    s = next(x for x in train_samples if x["label"]["tool_calls"] and len(x["context"]["installed_apps"]) > 3)
    apps = gold_apps(s["label"])
    picked = bank.select(s["audio_text"], set(s["context"]["installed_apps"]), k=4, exclude_id=s["id"], exclude_apps=apps)
    assert picked and all(not (gold_apps(p["label"]) & apps) for p in picked)


def test_ask_target_counts_as_a_gold_app():
    label = {"tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["contact_id"]}}]}
    assert gold_apps(label) == {"sms"}


def test_out_of_scope_examples_survive_any_exclusion(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    installed = set(train_samples[0]["context"]["installed_apps"])
    picked = bank.select("kể chuyện cười", installed, k=20, exclude_apps=installed)
    assert picked and all(not p["label"]["tool_calls"] for p in picked)


def test_extract_unseen_history_chains_same_app_read_and_write():
    from lifemate.agents.fewshot import extract_unseen_history
    cat = {
        "parking.list_sessions": {"name": "parking.list_sessions", "kind": "read", "parameters": {"properties": {}}},
        "parking.end_session": {"name": "parking.end_session", "kind": "write",
                                "parameters": {"properties": {"session_id": {"type": "string", "x-ref": "parking.sessions"}}}},
    }
    sample = {
        "id": "p1",
        "history": [
            {"role": "user", "audio": "u0.wav"},
            {"role": "assistant", "text": "Xe đang gửi ở Bến Ninh Kiều.",
             "tool_calls": [{"name": "parking.list_sessions", "arguments": {}}],
             "tool_results": [{"sessions": [{"session_id": "park_1"}]}]},
            {"role": "user", "audio": "u1.wav"},
            {"role": "assistant", "text": "Đã kết thúc lượt gửi xe.",
             "tool_calls": [{"name": "parking.end_session", "arguments": {"session_id": "park_1"}}],
             "tool_results": [{"ok": True}]},
        ],
    }
    docs = extract_unseen_history([sample], cat, known_apps={"alarm"})
    assert len(docs) == 1
    assert [c["name"] for c in docs[0]["label"]["tool_calls"]] == ["parking.list_sessions", "parking.end_session"]
    assert docs[0]["id"] == "p1"
    assert extract_unseen_history([sample], cat, known_apps={"parking"}) == []


def test_from_data_with_history_examples_on_train_mines_only_the_pool(train_samples):
    from lifemate.splits import load_splits

    root = default_data_root()
    with_hist = ExampleBank.from_data(root, "train", history_examples=True)
    pool = set(load_splits()["pool"])
    assert {d["id"] for d in with_hist.docs} <= pool


def _group_bank():
    docs = [
        _sample("d1", "gọi taxi ra ga", [{"name": "ride_hailing.book", "arguments": {}}]),
        _sample("d2", "đặt lịch họp sân bay chiều nay", [{"name": "calendar.create_event", "arguments": {}}]),
        _sample("d3", "đặt lịch khám sân bay", [{"name": "calendar.create_event", "arguments": {}}]),
    ]
    return ExampleBank(docs, {"d1", "d2", "d3"})


GROUPS = {"ride_hailing": "mobility", "calendar": "planning", "parking": "mobility"}
TOOLS = [{"name": "parking.reserve", "description": "Giữ chỗ đỗ xe tại bãi", "parameters": {}},
         {"name": "calendar.create_event", "description": "Tạo sự kiện", "parameters": {}}]


def test_group_slot_brings_an_example_of_the_unseen_apps_group():
    bank = _group_bank()
    installed = {"ride_hailing", "calendar", "parking"}
    plain = bank.select("đặt lịch giữ chỗ đỗ xe sân bay", installed, 2)
    assert {d["id"] for d in plain} == {"d2", "d3"}  # BM25 alone: no mobility example
    picked = bank.select("đặt lịch giữ chỗ đỗ xe sân bay", installed, 2, tools=TOOLS, group_slots=1, app_groups=GROUPS)
    assert [d["id"] for d in picked] == [plain[0]["id"], "d1"]  # d1 (ride_hailing, mobility like parking) takes the last slot


def test_group_slot_changes_nothing_without_an_unseen_app_or_a_group_example():
    bank = _group_bank()
    tools = [t for t in TOOLS if not t["name"].startswith("parking.")]
    kw = {"tools": tools, "group_slots": 1, "app_groups": GROUPS}
    assert bank.select("đặt lịch sân bay", {"ride_hailing", "calendar"}, 2, **kw) == \
        bank.select("đặt lịch sân bay", {"ride_hailing", "calendar"}, 2)
    assert bank.unseen_target({"đỗ", "xe"}, TOOLS) == "parking"
    no_mobility = {"calendar": "planning", "ride_hailing": "transport", "parking": "mobility"}
    assert bank.select("đặt lịch giữ chỗ đỗ xe", {"ride_hailing", "calendar", "parking"}, 2, tools=TOOLS, group_slots=1,
                       app_groups=no_mobility) == bank.select("đặt lịch giữ chỗ đỗ xe", {"ride_hailing", "calendar", "parking"}, 2)
