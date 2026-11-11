"""Correct the date of a call when the request holds exactly one relative date phrase.

The model sometimes writes the wrong calendar day for "thứ ba tuần sau" and the like. resolve_date turns the phrase into an
ISO date by calendar maths (weeks start on Monday, which is how train gold resolves them), and fix_call_dates puts that date
into the call's single date-valued string argument, keeping any time part. It changes an argument of a call the model sent
and never sends a new call. No schema or app name decides anything except the description of the argument (a past range
is left alone). Shared by the simulator (`lifemate.sim`, rewrite "date_fix") and the agent (`date_fix=True`).

weekdays_only=True is the narrower rule of `weekday_fix` (rewrite "weekday_fix", agent `weekday_fix=True`):
1. A day phrase (hôm nay, ngày mai, ngày mốt, sáng/trưa/chiều/tối mai) resolves to None, the model is left to it. It still
   counts as a phrase, so a request with a day phrase and a weekday phrase stays None.
2. A weekday phrase directly followed by giờ, rưỡi, phút or tiếng resolves to None (ASR hears "lúc hai giờ" as
   "thứ hai giờ").
3. After a qualified weekday (tuần sau, tuần này, này), "hôm đó" / "hôm ấy" later in the request name that same day, so
   they are not another time reference. A bare weekday keeps the rule above.
4. fix_call_dates fixes every date-valued string argument of the call when they all share one date part (a start and an
   end on the same day); arguments on different days, or any that names a past range, leave the call alone.
With weekdays_only=False every function behaves as it did before the option existed.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta

from .ask_guard import fold
from .dates import WEEKDAYS

_DAY_NAMES = {name: i for i, name in enumerate(WEEKDAYS)}
# One alternation, so overlapping phrases ("thứ ba tuần sau") count once. "mai" alone is also a name, so it only counts
# after a part of the day or after "ngày".
_PHRASE = re.compile(
    r"(?<!\w)(?:(?P<day>hôm nay|ngày mốt|ngày mai|(?:sáng|trưa|chiều|tối) mai)"
    r"|(?P<weekday>thứ (?:hai|ba|tư|năm|sáu|bảy)|chủ nhật)(?P<when> tuần sau| tuần này| này)?)(?!\w)")
_DATE_VALUE = re.compile(r"^\d{4}-\d{2}-\d{2}(?=$|[T ])")
# Any other time reference in the request makes the phrase ambiguous: the date may belong to another call or to a day the
# phrase does not name ("hôm nay tôi đi bộ, à mai nhắc tôi", "sáng nay ... tối mai", "mười một tháng mười").
_OTHER_TIME = re.compile(
    r"(?<!\w)(?:mai|nay|này|hôm qua|hôm kia|hôm trước|hôm đó|hôm ấy|bữa|tuần trước|tuần sau|tuần này|tháng|năm ngoái|"
    r"đầu tuần|cuối tuần|cuối tháng|thứ (?:hai|ba|tư|năm|sáu|bảy)|chủ nhật|ngày \w*\d\w*|\d+/\d+)(?!\w)")
# "thứ hai" is an ordinal after a counting word or before "trong danh sách": "quán thứ hai", "chuyến thứ ba"
_COUNTERS = {"cái", "quán", "chuyến", "bài", "lần", "cuốn", "món", "bộ", "phim", "tập", "người", "căn", "nhà", "vé", "dòng",
             "mục", "số", "bản", "video", "ứng", "viên", "lựa", "chọn", "phương", "án", "đáp", "địa", "chỉ", "tiệm", "hàng",
             "tin", "cuộc", "thư", "ảnh", "tệp", "hát", "album", "kết", "quả", "bác", "sĩ", "cô", "thầy", "bạn", "anh", "chị",
             "ông", "bà", "em", "khách", "sạn", "hãng", "đợt", "vòng", "trận", "ca", "buổi", "tiết", "hạng", "giải", "câu",
             "chương", "phần", "trang", "khóa", "lớp", "đơn", "hóa", "phòng", "tầng", "lô", "nhóm", "đội", "thẻ", "tài", "khoản"}
_ORDINAL_AFTER = re.compile(r"^\s*(?:trong|ở|của)\s+(?:danh sách|kết quả|các|những|số)")
_PAST_RANGE = re.compile(r"tu thoi diem|qua khu|da qua|truoc day|lich su|since")  # matched on folded words
_CLOCK_AFTER = re.compile(r"^\s*(?:giờ|rưỡi|phút|tiếng)(?!\w)")  # weekdays_only: "thứ hai giờ chiều" is a clock time
_SAME_DAY = re.compile(r"(?<!\w)hôm (?:đó|ấy)(?!\w)")  # weekdays_only: the day a qualified weekday already named


def resolve_date(request: str, current_time: str, weekdays_only: bool = False) -> str | None:
    """The ISO date of the request's only date phrase, or None (no phrase, several, or a day that cannot be told).

    hôm nay, ngày mai, ngày mốt, sáng/trưa/chiều/tối mai; thứ X tuần sau (that weekday of next week); thứ X này (this
    week, None when already past); thứ X without a qualifier and chủ nhật (the next one, today included).
    weekdays_only applies rules 1 to 3 of the module docstring."""
    matches = list(_PHRASE.finditer(unicodedata.normalize("NFC", request).lower()))  # offsets are those of `lowered`
    if len(matches) != 1:
        return None
    m = matches[0]
    lowered = unicodedata.normalize("NFC", request).lower()
    if weekdays_only and (m["day"] or _CLOCK_AFTER.match(lowered[m.end():])):
        return None
    after = lowered[m.end():]
    if weekdays_only and m["when"]:
        after = _SAME_DAY.sub(" ", after)
    if _OTHER_TIME.search(lowered[: m.start()] + " " + after):
        return None
    if m["weekday"] and m["weekday"].startswith("thứ"):
        before = lowered[: m.start()].split()
        if (before and before[-1] in _COUNTERS) or _ORDINAL_AFTER.match(lowered[m.end():]):
            return None
    today = date.fromisoformat(current_time[:10])  # the local date as written, no timezone conversion
    if m["day"]:
        offset = 0 if m["day"] == "hôm nay" else 2 if m["day"] == "ngày mốt" else 1
        return (today + timedelta(days=offset)).isoformat()
    monday = today - timedelta(days=today.weekday())
    this_week = monday + timedelta(days=_DAY_NAMES[m["weekday"]])
    when = (m["when"] or "").strip()
    if when == "tuần sau":
        return (this_week + timedelta(days=7)).isoformat()
    if when in ("tuần này", "này"):
        return this_week.isoformat() if this_week >= today else None
    return (this_week if this_week >= today else this_week + timedelta(days=7)).isoformat()


def _names_past_range(spec: dict | None, key: str) -> bool:
    prop = (((spec or {}).get("parameters") or {}).get("properties") or {}).get(key) or {}
    return bool(_PAST_RANGE.search(" ".join(fold(str(prop.get("description", ""))))))


def fix_call_dates(call: dict, spec: dict | None, request: str, current_time: str, weekdays_only: bool = False) -> dict:
    """A copy of `call` whose single date-valued string argument carries the date of the request's phrase, or `call`
    itself when nothing applies: no phrase, no such argument, two of them, or an argument that names a past range.
    With weekdays_only, several date-valued arguments are all fixed when they share one date part (rule 4)."""
    args = call.get("arguments", {})
    dated = [k for k, v in args.items() if isinstance(v, str) and _DATE_VALUE.match(v)]
    if not dated or any(_names_past_range(spec, k) for k in dated):
        return call
    if len(dated) != 1 and not (weekdays_only and len({args[k][:10] for k in dated}) == 1):
        return call
    new = resolve_date(request, current_time, weekdays_only)
    if new is None or args[dated[0]][:10] == new:
        return call
    return dict(call, arguments=dict(args, **{k: new + args[k][10:] for k in dated}))


def fix_calls(calls: list[dict], specs: dict[str, dict], request: str, current_time: str,
              weekdays_only: bool = False) -> list[dict]:
    """fix_call_dates over the calls of one turn, only when exactly one of them carries a date-valued string argument:
    with two, the phrase may date either (or neither), so nothing is changed. Returns the same list when nothing changes."""
    dated = [c for c in calls if any(isinstance(v, str) and _DATE_VALUE.match(v) for v in c.get("arguments", {}).values())]
    if len(dated) != 1:
        return calls
    fixed = [fix_call_dates(c, specs.get(c["name"]), request, current_time, weekdays_only) for c in calls]
    return calls if all(f is c for f, c in zip(fixed, calls)) else fixed
