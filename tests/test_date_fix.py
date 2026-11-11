import pytest

from lifemate.agents.date_fix import fix_call_dates, resolve_date

NOW = "2026-10-01T09:00:00+07:00"  # a Thursday; weeks start on Monday


@pytest.mark.parametrize("now, text, date", [
    ("2026-10-01T09:00:00", "đặt lịch thứ ba tuần sau", "2026-10-06"),
    ("2026-10-05T09:00:00", "thứ sáu tuần sau họp", "2026-10-16"),
    ("2026-12-04T09:00:00", "thứ sáu tuần sau họp", "2026-12-11"),
    ("2026-09-24T09:00:00", "vé máy bay thứ bảy tuần sau", "2026-10-03"),
])
def test_resolve_date_next_week(now, text, date):
    assert resolve_date(text, now) == date


@pytest.mark.parametrize("text, date", [
    ("nhắc tôi hôm nay", "2026-10-01"),
    ("hẹn ngày mai nhé", "2026-10-02"),
    ("báo thức sáng mai", "2026-10-02"),
    ("họp trưa mai", "2026-10-02"),
    ("đặt bàn chiều mai", "2026-10-02"),
    ("gọi lại tối mai", "2026-10-02"),
    ("đi chơi ngày mốt", "2026-10-03"),
    ("họp thứ sáu này", "2026-10-02"),
    ("họp thứ sáu", "2026-10-02"),          # no qualifier: the next one, today included
    ("họp thứ năm", "2026-10-01"),
    ("họp thứ hai", "2026-10-05"),          # Monday of this week is past, so the coming one
    ("chủ nhật này đi bơi", "2026-10-04"),
    ("chủ nhật tuần sau đi bơi", "2026-10-11"),
    ("Họp Thứ Ba Tuần Sau", "2026-10-06"),  # case is ignored
])
def test_resolve_date_phrases(text, date):
    assert resolve_date(text, NOW) == date


def test_a_past_day_of_this_week_is_not_guessed():
    assert resolve_date("thứ hai này", NOW) is None


def test_mai_as_a_name_is_not_a_date():
    assert resolve_date("gọi cho cô Mai", "2026-10-01T09:00:00") is None
    assert resolve_date("nhắn cho anh mai là mình đến", NOW) is None


def test_two_phrases_are_left_alone():
    assert resolve_date("hôm nay xem giá, sáng mai nhắc tôi", "2026-10-01T09:00:00") is None


def test_no_phrase_is_none():
    assert resolve_date("bật đèn phòng khách", NOW) is None


SPEC = {"parameters": {"properties": {"start": {"type": "string", "description": "ISO-8601, ví dụ 2026-11-11T05:30:00+07:00"},
                                      "title": {"type": "string"}}}}


def call(**args):
    return {"name": "calendar.create_event", "arguments": args}


def test_fix_keeps_time_part():
    fixed = fix_call_dates(call(start="2026-09-30T15:00:00+07:00", title="họp"), SPEC, "họp chiều mai", NOW)
    assert fixed["arguments"] == {"start": "2026-10-02T15:00:00+07:00", "title": "họp"}


def test_fix_changes_a_plain_date():
    fixed = fix_call_dates(call(start="2026-10-09"), SPEC, "họp thứ sáu tuần sau", NOW)
    assert fixed["arguments"]["start"] == "2026-10-09"
    fixed = fix_call_dates(call(start="2026-10-08"), SPEC, "họp thứ sáu tuần sau", NOW)
    assert fixed["arguments"]["start"] == "2026-10-09"


def test_fix_does_not_change_the_input():
    original = call(start="2026-09-30T15:00:00+07:00")
    fix_call_dates(original, SPEC, "họp ngày mai", NOW)
    assert original["arguments"]["start"] == "2026-09-30T15:00:00+07:00"


def test_fix_skips_two_date_arguments():
    spec = {"parameters": {"properties": {"start": {"type": "string"}, "end": {"type": "string"}}}}
    c = call(start="2026-09-30T15:00:00+07:00", end="2026-09-30T16:00:00+07:00")
    assert fix_call_dates(c, spec, "họp ngày mai", NOW) == c


def test_fix_skips_a_past_range_description():
    spec = {"parameters": {"properties": {"since": {"type": "string", "description": "Chỉ lấy cảnh báo từ thời điểm này"}}}}
    c = {"name": "home_security.list_alerts", "arguments": {"since": "2026-09-30T00:00:00+07:00"}}
    assert fix_call_dates(c, spec, "cảnh báo từ hôm nay", NOW) == c


def test_fix_leaves_calls_without_a_date_or_phrase_alone():
    c = call(title="họp")
    assert fix_call_dates(c, SPEC, "họp ngày mai", NOW) == c
    c = call(start="2026-09-30T15:00:00+07:00")
    assert fix_call_dates(c, SPEC, "họp lúc ba giờ", NOW) == c


def test_fix_ignores_time_only_and_non_string_values():
    spec = {"parameters": {"properties": {"time": {"type": "string"}, "n": {"type": "integer"}}}}
    c = {"name": "alarm.create", "arguments": {"time": "07:30", "n": 3}}
    assert fix_call_dates(c, spec, "báo thức ngày mai", NOW) == c


def test_fix_works_without_a_spec():
    fixed = fix_call_dates(call(start="2026-09-30"), None, "ngày mai", NOW)
    assert fixed["arguments"]["start"] == "2026-10-02"


# --- false positives found in review ----------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "hôm nay tôi đi bộ ba lăm phút, à mai tám giờ sáng nhắc tôi uống thuốc",   # a second time reference ("mai")
    "ghi buổi yoga từ sáu rưỡi sáng nay, rồi bảy giờ tối mai nhắc em uống thuốc",
    "buổi đầu tám giờ sáng thứ hai tuần sau, hôm trước em đã đăng ký",
    "thứ ba nhắc anh, mười một tháng mười là ngày khám",
    "tám giờ tối nay, với chủ nhật nữa",
    "mai với thứ bảy em học nhóm",
    "sớm mai sáu giờ đón đoàn, chuông tối thứ năm là shipper",
    "bữa lên Bà Nà trời mưa không, tám giờ tối thứ ba nhắc anh",
])
def test_a_second_time_reference_gives_no_date(text):
    assert resolve_date(text, NOW) is None


@pytest.mark.parametrize("text", [
    "chọn quán thứ hai giúp anh",
    "đặt chuyến thứ hai trong danh sách",
    "mở bài thứ ba",
    "lần thứ năm rồi đó",
])
def test_an_ordinal_is_not_a_weekday(text):
    assert resolve_date(text, NOW) is None


def test_a_real_weekday_after_a_non_counter_still_resolves():
    assert resolve_date("họp lúc ba giờ chiều thứ sáu tuần sau", NOW) == "2026-10-09"
    assert resolve_date("nhắc tôi uống thuốc ngày mai", NOW) == "2026-10-02"


def test_fix_calls_changes_nothing_when_two_calls_carry_a_date():
    from lifemate.agents.date_fix import fix_calls

    two = [call(start="2026-09-30T15:00:00+07:00"), {"name": "tasks.add", "arguments": {"due": "2026-09-30T18:00:00+07:00"}}]
    assert fix_calls(two, {"calendar.create_event": SPEC}, "họp ngày mai", NOW) == two
    one = fix_calls([call(start="2026-09-30T15:00:00+07:00"), {"name": "x.read", "arguments": {}}], {}, "họp ngày mai", NOW)
    assert one[0]["arguments"]["start"] == "2026-10-02T15:00:00+07:00" and one[1] == {"name": "x.read", "arguments": {}}


def test_the_fix_never_changes_a_gold_call_of_train(train_samples):
    """Regression scan: on the gold calls of all train samples the fix may only touch these known semantic cases, where
    the date of a call belongs to something the request does not date by a phrase (a draw date, a trip date)."""
    from lifemate.data import default_data_root, load_catalog
    from lifemate.agents.date_fix import fix_calls

    catalog = load_catalog(default_data_root(), "train")
    changed = {s["id"] for s in train_samples
               if fix_calls(s["label"]["tool_calls"], catalog, s["audio_text"], s["context"]["current_time"])
               != s["label"]["tool_calls"]}
    assert changed <= {"train_1249", "train_1338"}, sorted(changed)


# weekdays_only (weekday_fix): only "thứ X" phrases, a clock word after the weekday, "hôm đó" after a qualified weekday,
# and every date argument of a call fixed together when they share one day.
THU = "2026-09-24T09:00:00+07:00"   # a Thursday


def test_weekday_next_week():
    assert resolve_date("Đặt cho tôi một ghế chuyến RV21 đi Đà Nẵng thứ bảy tuần sau", THU, weekdays_only=True) == "2026-10-03"


def test_same_day_reference_after_a_qualified_weekday_is_allowed():
    text = "Thứ tư tuần sau anh đi Hải Phòng, giữ một chỗ gửi xe từ năm giờ sáng hôm đó"
    assert resolve_date(text, THU, weekdays_only=True) == "2026-09-30"
    assert resolve_date(text, THU) is None            # the old rule is unchanged


def test_same_day_reference_after_a_bare_weekday_still_blocks():
    assert resolve_date("Thứ hai họp, nhắc anh sáng hôm đó", THU, weekdays_only=True) is None


def test_weekday_followed_by_a_clock_word_is_not_a_weekday():
    assert resolve_date("ghi vào lịch thứ hai giờ chiều", "2026-10-05T09:00:00+07:00", weekdays_only=True) is None


def test_day_phrases_are_left_to_the_model():
    assert resolve_date("hủy hẹn sáng mai", "2026-10-20T21:10:00+07:00", weekdays_only=True) is None
    assert resolve_date("hủy hẹn sáng mai", "2026-10-20T21:10:00+07:00") == "2026-10-21"


def test_ordinal_is_not_a_weekday():
    assert resolve_date("mở quán thứ hai trong danh sách", THU, weekdays_only=True) is None


EVENT = "tạo lịch Bàn giao lúc mười giờ sáng thứ tư tuần sau"
NOW = "2026-10-01T09:00:00+07:00"


def test_start_and_end_on_one_day_are_fixed_together():
    call = {"name": "calendar.create_event", "arguments": {
        "title": "Bàn giao", "start": "2026-10-08T10:00:00+07:00", "end": "2026-10-08T11:00:00+07:00"}}
    fixed = fix_call_dates(call, None, EVENT, NOW, weekdays_only=True)
    assert fixed["arguments"]["start"] == "2026-10-07T10:00:00+07:00"
    assert fixed["arguments"]["end"] == "2026-10-07T11:00:00+07:00"
    assert fix_call_dates(call, None, EVENT, NOW) is call          # the old rule needs a single dated argument


def test_dates_on_different_days_are_left_alone():
    call = {"name": "parking.reserve", "arguments": {"start": "2026-10-08T05:00:00+07:00", "end": "2026-10-11T05:00:00+07:00"}}
    assert fix_call_dates(call, None, EVENT, NOW, weekdays_only=True) is call


def test_a_right_date_returns_the_same_call():
    call = {"name": "tasks.add", "arguments": {"title": "x", "due": "2026-10-07T10:00:00+07:00"}}
    assert fix_call_dates(call, None, EVENT, NOW, weekdays_only=True) is call
