from lifemate.scoring import call_matches, count_matches, score_sample, token_f1

FUZZY = {"sms.send": {"content"}}


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def test_token_f1_is_case_insensitive():
    assert token_f1("Mai Chạy Bộ", "mai chạy bộ") == 1.0
    assert token_f1("a b", "c d") == 0.0


def test_fuzzy_argument_needs_token_f1_above_threshold():
    gold = [call("sms.send", contact_id="c_08", content="mai chạy bộ dời sang 09:00 nhé")]
    close = [call("sms.send", contact_id="c_08", content="mai chạy bộ dời sang 09:00")]
    far = [call("sms.send", contact_id="c_08", content="hủy họp")]
    assert count_matches(close, gold, FUZZY) == 1
    assert count_matches(far, gold, FUZZY) == 0


def test_non_fuzzy_argument_must_be_identical():
    gold = [call("alarm.create", time="2026-11-17T06:30:00+07:00")]
    assert count_matches([call("alarm.create", time="2026-11-17T06:30:00+07:00")], gold, {}) == 1
    assert count_matches([call("alarm.create", time="2026-11-17T06:31:00+07:00")], gold, {}) == 0


def test_each_call_matches_at_most_once():
    gold = [call("smart_home.get_status")]
    pred = [call("smart_home.get_status"), call("smart_home.get_status")]
    r = score_sample(pred, "h", gold, "h", {})
    assert r["matched"] == 1 and abs(r["score"] - 2 / 3) < 1e-9  # P = 1/2, R = 1


def test_hash_mismatch_scores_zero_but_keeps_f1():
    gold = [call("smart_home.get_status")]
    r = score_sample(gold, "x", gold, "y", {})
    assert r["score"] == 0.0 and r["f1"] == 1.0


def test_out_of_scope_needs_empty_submission():
    assert score_sample([], "h", [], "h", {})["score"] == 1.0
    assert score_sample([call("weather.get_current")], "h", [], "h", {})["score"] == 0.0


def test_extra_optional_argument_still_matches_but_a_missing_one_does_not():
    gold = {"name": "appliances.list", "arguments": {}}
    assert call_matches({"name": "appliances.list", "arguments": {"type": "dishwasher"}}, gold, set())
    gold = {"name": "bike_rental.find_bikes", "arguments": {"near": "Lê Văn Sỹ"}}
    assert not call_matches({"name": "bike_rental.find_bikes", "arguments": {}}, gold, set())


def test_ask_params_compare_as_a_set():
    gold = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount", "beneficiary_id"]}}
    swapped = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["beneficiary_id", "amount"]}}
    extra = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount", "beneficiary_id", "note"]}}
    fewer = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "banking.transfer", "params": ["amount"]}}
    assert call_matches(swapped, gold, set())
    assert not call_matches(extra, gold, set())
    assert not call_matches(fewer, gold, set())
