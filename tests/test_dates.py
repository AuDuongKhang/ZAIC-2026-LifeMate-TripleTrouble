from lifemate.agents.dates import date_table


def lines(current_time: str) -> list[str]:
    return date_table(current_time).splitlines()


def test_thursday_table():
    out = lines("2026-10-01T09:00:00+07:00")
    assert out[0] == "Calendar (weeks start on Monday):"
    assert out[1] == "2026-10-01 thứ năm (hôm nay, tuần này)"
    assert out[2] == "2026-10-02 thứ sáu (ngày mai, tuần này)"
    assert out[3] == "2026-10-03 thứ bảy (ngày mốt, tuần này)"
    assert out[4] == "2026-10-04 chủ nhật (tuần này)"
    assert "2026-10-06 thứ ba (tuần sau)" in out
    assert out[-1] == "2026-10-11 chủ nhật (tuần sau)"
    assert len(out) - 1 == 11  # day lines


def test_friday_next_week():
    assert "2026-12-11 thứ sáu (tuần sau)" in lines("2026-12-04T08:00:00+07:00")


def test_sunday_rolls_to_next_week():
    out = lines("2026-09-20T10:00:00+07:00")
    assert out[1] == "2026-09-20 chủ nhật (hôm nay, tuần này)"
    assert out[2] == "2026-09-21 thứ hai (ngày mai, tuần sau)"
    assert out[-1] == "2026-09-27 chủ nhật (tuần sau)"
