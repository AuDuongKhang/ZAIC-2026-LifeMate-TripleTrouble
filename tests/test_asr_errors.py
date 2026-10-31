from lifemate.asr.wer import normalize, wer
from lifemate.asr_errors import align, analyze, gold_tokens, word_category


def test_word_categories():
    assert word_category("9", False) == "number" and word_category("năm", False) == "number"
    assert word_category("SJC", False) == "foreign" and word_category("VTV6", False) == "foreign"
    assert word_category("Netflix", False) == "foreign" and word_category("wifi,", False) == "foreign"
    assert word_category("Minh", False) == "name" and word_category("Trợ", True) == "other"
    assert word_category("đèn", False) == "other" and word_category("Ba", False) == "name"


def test_ambiguous_number_words_need_a_number_neighbour():
    cats = gold_tokens("Không bật đèn, gọi số không chín một hai và mua một cái.")
    assert [c for t, c in cats if t == "không"] == ["other", "number"]
    assert [c for t, c in cats if t == "một"] == ["number", "other"]


def test_gold_tokens_match_normalize():
    ref = "Nếu giá vàng SJC dưới 123 triệu thì 9 giờ nhắc Sếp Dũng nhé."
    assert [t for t, _ in gold_tokens(ref)] == normalize(ref)
    cats = dict(gold_tokens(ref))
    assert cats["sjc"] == "foreign" and cats["dũng"] == "name" and cats["triệu"] == "number"


def test_align_counts_match_wer():
    ref, hyp = normalize("giảm độ sáng đèn phòng ngủ"), normalize("giảm sáng điện phòng ngủ nhỏ")
    ops = align(ref, hyp)
    errors = sum(op != "ok" for op, _, _ in ops)
    assert errors / len(ref) == wer("giảm độ sáng đèn phòng ngủ", "giảm sáng điện phòng ngủ nhỏ")


def test_analyze_attributes_errors_to_categories():
    refs = {"a.wav": "Nhắn Minh mua vàng SJC lúc 9 giờ.", "a_u1.wav": "Bật đèn."}
    rows = {"a.wav": {"text": "nhắn minh mua vàng stc lúc chín giờ"}, "a_u1.wav": {"text": "bật điện"}}
    res = analyze(rows, refs)
    assert res["by_cat"]["foreign"]["errors"] == 1 and res["by_cat"]["number"]["errors"] == 0
    assert res["subs"][("sjc", "stc")] == 1 and res["subs"][("đèn", "điện")] == 1
    assert {c["history"] for c in res["clips"]} == {True, False}
