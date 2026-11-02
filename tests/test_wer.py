import pytest

from lifemate.asr.wer import cer, mean_wer, normalize, number_words, wer


def test_wer_ignores_case_and_punctuation():
    assert wer("Tắt đèn phòng khách.", "tắt đèn phòng khách") == 0.0


def test_wer_counts_substitution_and_deletion():
    assert wer("chuyển cho mẹ năm trăm nghìn", "chuyển cho mẹ năm nghìn") == 1 / 6


def test_wer_nfc():
    assert wer("hoà", "hoà") == 0.0


def test_wer_empty_reference():
    assert wer("", "") == 0.0 and wer("", "xin chào") == 1.0


def test_mean_wer_skips_traces_without_request():
    traces = [{"id": "a", "request": "bật đèn"}, {"id": "b", "request": None}]
    samples = {"a": {"audio_text": "Bật đèn."}, "b": {"audio_text": "tắt quạt"}}
    assert mean_wer(traces, samples) == 0.0


@pytest.mark.parametrize("n,words", [(5, "năm"), (10, "mười"), (15, "mười lăm"), (21, "hai mươi mốt"), (25, "hai mươi lăm"),
    (105, "một trăm lẻ năm"), (1000, "một nghìn"), (2800000, "hai triệu tám trăm nghìn"), (0, "không"),
    (1005, "một nghìn không trăm lẻ năm"), (123000000, "một trăm hai mươi ba triệu")])
def test_number_words(n, words):
    assert number_words(n) == words


def test_digits_match_spelled_numbers():
    assert wer("Đặt báo thức 5 giờ", "đặt báo thức năm giờ") == 0.0
    assert wer("nạp 50 ngàn", "nạp năm mươi nghìn") == 0.0


def test_leading_zero_digits_are_read_one_by_one():
    assert normalize("số 0905") == ["số", "không", "chín", "không", "năm"]


def test_digits_inside_words_are_kept():
    assert normalize("kênh S+ Sport 4G") == ["kênh", "s", "sport", "4g"]


def test_old_and_new_tone_placement_match():
    assert wer("Hủy gói, khóa cửa, bật điều hòa", "huỷ gói khoá cửa bật điều hoà") == 0.0
    assert normalize("hoàn thành quý") == ["hoàn", "thành", "quý"]  # closed syllables and qu- untouched


def test_colloquial_tens_match_digits():
    assert wer("Xăng 95 bao nhiêu", "xăng chín lăm bao nhiêu") == 0.0
    assert wer("21 người", "hai mốt người") == 0.0
    assert normalize("mai mốt") == ["mai", "mốt"] and normalize("mười lăm") == ["mười", "lăm"]


def test_cer():
    assert cer("đèn", "đen") == 1 / 3
