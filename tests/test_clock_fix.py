import pytest

from lifemate.agents.clock_fix import clocks, fix_call_clock, fix_calls_clock, has_datetime

NOW = "2026-11-05T11:40:00+07:00"
EVENING = "Nhớ nhắc mình gọi lại cho khách lúc tám rưỡi tối mai nhé."


def spoken(text):
    return [(c["hour"], c["minute"], c["part"]) for c in clocks(text)]


def call(**arguments):
    return {"name": "reminders.create", "arguments": arguments}


@pytest.mark.parametrize("text, expected", [
    (EVENING, [(8, 30, "tối")]),
    ("rồi mười giờ mười lăm nhắc tôi ra lấy đồ nhé", [(10, 15, None)]),
    ("đón lúc bốn giờ bốn lăm", [(4, 45, None)]),
    ("học từ chín giờ tới mười giờ kém mười lăm", [(9, 0, None), (9, 45, None)]),
    ("Đặt báo thức lúc 8 giờ sáng mai", [(8, 0, "sáng")]),
    ("sáng mai lúc bảy giờ gọi tôi dậy", [(7, 0, "sáng")]),
    ("chiều thứ sáu lúc ba giờ họp nhóm", [(3, 0, "chiều")]),
    ("tối nay bảy giờ rưỡi đón con", [(7, 30, "tối")]),
    ("Mai với thứ bảy, ba rưỡi chiều, em học nhóm", [(3, 30, "chiều")]),
    ("đi bộ từ bảy sáng đến năm rưỡi chiều", [(7, 0, "sáng"), (5, 30, "chiều")]),
    ("họp lúc 14:30", [(14, 30, None)]),
    ("20h30 tối nay", [(20, 30, "tối")]),
    ("nhắc tôi sau hai giờ nữa", []),
    ("chạy máy trong hai giờ đồng hồ", []),
    ("khoảng tám giờ tối ghé nhà", []),
    ("bây giờ bật đèn giúp tôi", []),
    ("mua hai mươi quả trứng", []),
    ("thứ hai tối họp nhóm", []),
])
def test_clocks(text, expected):
    assert spoken(text) == expected


def test_a_day_part_sets_the_time_of_day():
    c = call(content="gọi lại cho khách", time="2026-11-06T19:30:00+07:00")
    assert fix_call_clock(c, EVENING, NOW) == call(content="gọi lại cho khách", time="2026-11-06T20:30:00+07:00")
    assert c["arguments"]["time"] == "2026-11-06T19:30:00+07:00"  # the input is not mutated


def test_a_right_time_and_a_call_without_a_datetime_are_returned_as_they_are():
    right, dated = call(time="2026-11-06T20:30:00+07:00"), call(date="2026-11-06")
    assert fix_call_clock(right, EVENING, NOW) is right
    assert fix_call_clock(dated, EVENING, NOW) is dated
    assert has_datetime(right) and not has_datetime(dated)


def test_a_start_and_end_pair_moves_together():
    c = call(title="Học nhóm", start="2026-11-27T18:30:00+07:00", end="2026-11-27T19:30:00+07:00")
    fixed = fix_call_clock(c, "Mai ba rưỡi chiều em học nhóm ở Cao Nguyên, ghi lịch nha.", "2026-11-26T08:00:00+07:00")
    assert fixed["arguments"] == {"title": "Học nhóm", "start": "2026-11-27T15:30:00+07:00", "end": "2026-11-27T16:30:00+07:00"}


def test_an_end_pushed_past_midnight_moves_to_the_next_day():
    c = call(start="2026-10-01T10:00:00+07:00", end="2026-10-01T11:00:00+07:00")
    fixed = fix_call_clock(c, "họp lúc mười một rưỡi tối nay", "2026-10-01T09:00:00+07:00")
    assert fixed["arguments"] == {"start": "2026-10-01T23:30:00+07:00", "end": "2026-10-02T00:30:00+07:00"}


def test_the_shape_of_the_value_is_kept():
    assert fix_call_clock(call(time="2026-11-06T19:30"), EVENING, NOW) == call(time="2026-11-06T20:30")


def test_three_datetime_arguments_are_left_alone():
    c = call(a="2026-11-06T19:30:00+07:00", b="2026-11-06T19:40:00+07:00", c="2026-11-06T19:50:00+07:00")
    assert fix_call_clock(c, EVENING, NOW) is c


@pytest.mark.parametrize("request_text, value, now", [
    # a time the model had to compute
    ("Chín giờ sáng mai anh phải có mặt ở Bệnh viện Bạch Mai, nhắc anh ra khỏi nhà sao cho tới sớm mười lăm phút nhé.",
     "2026-09-22T08:30:00+07:00", "2026-09-21T09:00:00+07:00"),
    ("Tám giờ sáng thứ hai tuần sau anh đi xe buýt, đặt sẵn taxi đón anh đúng lúc xe tới nha.",
     "2026-10-12T08:35:00+07:00", "2026-10-05T09:00:00+07:00"),
    # two clock times
    ("Sáng nay lịch chị trống thì xếp đi chợ lúc tám giờ, không thì ba giờ chiều.",
     "2026-10-31T15:00:00+07:00", "2026-10-31T06:45:00+07:00"),
    ("tạo thêm cái lịch đi bộ từ bảy sáng đến năm rưỡi chiều hôm ấy nhé", "2026-10-16T07:00:00+07:00", "2026-10-10T09:00:00+07:00"),
    # no rule for midnight
    ("mười hai giờ đêm nay gọi tôi", "2026-10-01T23:00:00+07:00", "2026-10-01T09:00:00+07:00"),
])
def test_requests_that_change_nothing(request_text, value, now):
    c = call(time=value)
    assert fix_call_clock(c, request_text, now) is c


def test_a_bare_clock_moves_an_evening_time_to_the_morning_still_ahead():
    request = "bật chế độ giặt nhanh giúp tôi, rồi mười giờ mười lăm nhắc tôi ra lấy đồ nhé."
    fixed = fix_call_clock(call(time="2026-10-01T22:15:00+07:00"), request, "2026-10-01T09:00:00+07:00")
    assert fixed == call(time="2026-10-01T10:15:00+07:00")


@pytest.mark.parametrize("request_text, value, now", [
    ("Năm giờ nhắc chị gọi Minh nhé", "2026-10-01T17:00:00+07:00", "2026-10-01T15:10:00+07:00"),  # 05:00 is already past
    ("Năm giờ nhắc chị gọi Minh nhé", "2026-10-01T05:00:00+07:00", "2026-10-01T15:10:00+07:00"),  # morning is never moved
    ("mười giờ mười lăm nhắc tôi ra lấy đồ nhé", "2026-10-02T22:15:00+07:00", "2026-10-01T09:00:00+07:00"),  # another day
    ("mười giờ mười lăm nhắc tôi ra lấy đồ nhé", "2026-10-01T22:00:00+07:00", "2026-10-01T09:00:00+07:00"),  # other minutes
])
def test_a_bare_clock_is_left_alone_otherwise(request_text, value, now):
    c = call(time=value)
    assert fix_call_clock(c, request_text, now) is c


def test_only_the_first_dated_call_of_a_list_is_fixed():
    read = {"name": "contacts.search", "arguments": {}}
    first, second = call(time="2026-11-06T19:30:00+07:00"), call(time="2026-11-06T19:00:00+07:00")
    fixed = fix_calls_clock([read, first, second], EVENING, NOW)
    assert fixed[0] is read and fixed[2] is second
    assert fixed[1] == call(time="2026-11-06T20:30:00+07:00")
    unchanged = [read, call(time="2026-11-06T20:30:00+07:00"), second]
    assert fix_calls_clock(unchanged, EVENING, NOW) is unchanged


@pytest.mark.parametrize("value, now", [
    ("2026-11-31T19:30:00+07:00", NOW),  # a day the calendar does not have
    ("2026-11-06T19:30:00+07:00", "not a time"),
])
def test_a_value_that_is_not_a_real_datetime_is_left_alone(value, now):
    c = call(time=value)
    assert fix_call_clock(c, EVENING, now) is c


def test_a_day_part_before_a_digital_time_counts():
    assert spoken("tối nay 8h30 nhắc tôi gọi mẹ") == [(8, 30, "tối")]
    c = call(time="2026-10-01T20:30:00+07:00")
    assert fix_call_clock(c, "tối nay 8h30 nhắc tôi gọi mẹ", "2026-10-01T07:00:00+07:00") is c
