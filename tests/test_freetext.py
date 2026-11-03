from lifemate.agents.freetext import complete_call, complete_free_text, freetext_rewrite
from lifemate.sim import REWRITES

TITLE = "Logistics xanh: kho hàng Bắc Ninh giảm phát thải nhờ xe tải điện"

SPEC = {"parameters": {"properties": {
    "list_id": {"type": "string", "x-ref": "lists.search.id"},
    "title": {"type": "string", "x-match": "fuzzy"},
}}}


def test_prefix_is_completed():
    assert complete_free_text("Đọc bài logistics xanh", [TITLE]) == "Đọc bài " + TITLE


def test_value_already_full_is_kept():
    value = "Đọc bài " + TITLE
    assert complete_free_text(value, [TITLE]) == value


def test_one_token_overlap_is_not_enough():
    assert complete_free_text("Gọi Minh", ["Minh Anh"]) == "Gọi Minh"
    assert complete_free_text("Gọi Minh", ["Minh Anh"], min_tokens=1) == "Gọi Minh Anh"


def test_no_overlap_is_kept():
    value = "Đọc bài mới nhất về điện mặt trời"
    assert complete_free_text(value, ["Mái nhà dùng pin mặt trời"]) == value


def test_longest_overlap_then_first_text_wins():
    texts = ["xanh lá cây và biển", "logistics xanh lá cây rất nhiều", "logistics xanh lá mạ"]
    assert complete_free_text("Đọc logistics xanh lá", texts) == "Đọc logistics xanh lá cây rất nhiều"


def test_only_fuzzy_params_change():
    call = {"name": "tasks.add", "arguments": {"list_id": "l1", "title": "Đọc bài logistics xanh"}}
    out = complete_call(call, SPEC, [TITLE])
    assert out == {"name": "tasks.add", "arguments": {"list_id": "l1", "title": "Đọc bài " + TITLE}}
    assert call["arguments"]["title"] == "Đọc bài logistics xanh"  # never mutates


def test_call_without_spec_is_kept():
    call = {"name": "x.y", "arguments": {"title": "Đọc bài logistics xanh"}}
    assert complete_call(call, None, [TITLE]) == call


def test_rewrite_uses_results_seen_before_each_call():
    catalog = {"tasks.add": SPEC, "news.search": {"parameters": {"properties": {}}}}
    trace = {"id": "t", "request": "r", "calls": [
        {"name": "tasks.add", "arguments": {"title": "Đọc bài logistics xanh"}, "result": {"ok": True}},
        {"name": "news.search", "arguments": {}, "result": {"articles": [{"id": "a1", "title": TITLE}]}},
        {"name": "tasks.add", "arguments": {"title": "Đọc bài logistics xanh"}, "result": {"ok": True}},
    ]}
    calls = freetext_rewrite()(trace, catalog)
    assert [c["arguments"].get("title") for c in calls] == ["Đọc bài logistics xanh", None, "Đọc bài " + TITLE]
    assert freetext_rewrite()({"id": "t", "calls": trace["calls"][:1]}, catalog) is None


def test_registered_in_sim():
    assert REWRITES["freetext"] is freetext_rewrite
