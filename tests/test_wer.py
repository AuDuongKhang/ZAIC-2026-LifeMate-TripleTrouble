from lifemate.asr.wer import mean_wer, wer


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
