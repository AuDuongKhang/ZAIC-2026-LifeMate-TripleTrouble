"""Correct the time of day of a call when the request states exactly one clock time.

The model sometimes writes another time of day than the one the request states: "tám rưỡi tối mai" written as 19:30,
"ba rưỡi chiều" as 18:30, a bare "mười giờ mười lăm" said at 09:00 written as 22:15. date_fix and weekday_fix only correct
the calendar day. clocks lists the clock expressions of a request (spoken numbers, "rưỡi", "kém", "8 giờ", "20h30",
"bảy sáng") with the day part that goes with each, and fix_call_clock puts the stated time of day into the call:

1. Nothing changes when the request holds an offset word (sớm, trước giờ, đúng lúc, ...: the model had to compute the
   time), when it holds no clock or more than one, or when the call has no datetime argument or more than two.
2. With a day part, the target is the spoken hour on the 24-hour clock (sáng: as spoken below 12; trưa: +12 below 10;
   chiều, tối: +12 below 12; đêm, khuya: +12 from 9 to 11; a spoken hour of 13 or more as it is). Other cases have none.
3. Without a day part, an evening time on today's date that is the spoken time + 12 hours moves to the morning, but only
   while that morning time is still ahead of the current time.
4. Every datetime argument of the call shifts by the same amount (a start/end pair keeps its length; an end can move to
   the next day), on the wall clock as written, keeping the rest of the value (seconds, UTC offset) as it was.
fix_calls_clock applies this to the first call of a list that has a datetime argument only. 171 of 173 train requests
with one clock and a day part resolve to a gold time (the 2 misses are "arrive by" requests, excluded by the offset words).
It changes an argument of a call the model sent and never sends a new call. Shared by the simulator (`lifemate.sim`,
rewrite "clock_fix") and the agent (`clock_fix=True`).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta

_UNITS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "nhăm": 5, "sáu": 6,
          "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
_UNIT = "(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|bẩy|tám|chín)"
_NUM = rf"(?:\d{{1,2}}|mười(?: {_UNIT})?|{_UNIT} mươi(?: {_UNIT})?|(?:hai|ba|bốn|năm) (?:mốt|tư|lăm|nhăm)|{_UNIT})"  # 0..59
_PART = "(?:sáng|trưa|chiều|tối|đêm|khuya)"
_CLOCK = re.compile(
    rf"(?<!\w)(?P<h>{_NUM}) (?:giờ(?: (?P<half>rưỡi)| kém (?P<kem>{_NUM})| (?P<m>{_NUM})(?: phút)?)?"
    rf"(?! đồng hồ)(?! nữa)(?! \w+ nữa)|(?P<half2>rưỡi))(?!\w)")
_SHORT = re.compile(rf"(?<!\w)(?<!thứ )(?P<h>{_NUM}) (?P<p>sáng|trưa|chiều|tối)(?!\w)")  # "bảy sáng": no "giờ"
_DIGITAL = re.compile(r"(?<!\w)(?P<h>\d{1,2})(?:h|:|g)(?P<m>\d{2})?(?!\w)")
_PART_AFTER = re.compile(rf"^ (?:đúng )?(?P<p>{_PART})(?!\w)")
_PART_BEFORE = re.compile(rf"(?<!\w)(?P<p>{_PART})(?: (?:nay|mai|mốt|ni|này|hôm nay|ngày mai|thứ \w+|chủ nhật))?"
                          rf"(?: tuần sau)?(?: (?:lúc|vào|vào lúc|tầm|khoảng|cỡ|chừng))?$")
_DURATION_BEFORE = re.compile(r"(?:trong|mất|khoảng|thêm|nữa|cách|mỗi|suốt) $")
_OFFSET = re.compile(r"(?<!\w)(?:sao cho|sớm|muộn|trễ|trừ|kịp|tiếng|nửa giờ|trước (?:giờ|khi|lúc|đó|hạn)|sau khi|xong thì"
                     r"|cùng giờ|đúng giờ|đúng lúc|cùng lúc|theo giờ|giờ đó|giờ đấy|giờ ấy)(?!\w)")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def _number(phrase: str) -> int:
    """The value of a _NUM phrase: digits; mười [u] = 10 + u; u mươi [v] = 10u + v; two units = 10u + v; one unit."""
    if phrase.isdigit():
        return int(phrase)
    words = phrase.split()
    if words[0] == "mười":
        return 10 + (_UNITS[words[1]] if len(words) > 1 else 0)
    if len(words) > 1 and words[1] == "mươi":
        return 10 * _UNITS[words[0]] + (_UNITS[words[2]] if len(words) > 2 else 0)
    if len(words) == 2:
        return 10 * _UNITS[words[0]] + _UNITS[words[1]]
    return _UNITS[words[0]]


def _part(text: str, start: int, end: int, before: bool = True) -> str | None:
    after = _PART_AFTER.match(text[end:])
    if after:
        return after["p"]
    if before:
        found = _PART_BEFORE.search(text[:start].rstrip())
        if found:
            return found["p"]
    return None


def clocks(request: str) -> list[dict]:
    """Every clock expression of the request in text order: {"hour", "minute", "part"}, the hour as spoken (before the day
    part is applied), part one of sáng, trưa, chiều, tối, đêm, khuya or None."""
    text = unicodedata.normalize("NFC", request).lower()
    text = re.sub(r" +", " ", re.sub(r"[,.?!]", " ", text))
    found: list[tuple[int, dict]] = []
    spans: list[tuple[int, int]] = []

    def inside(position: int) -> bool:
        return any(start <= position < end for start, end in spans)

    clock_spans = []
    for m in _CLOCK.finditer(text):
        clock_spans.append(m.span())
        hour = _number(m["h"])
        if hour > 24:
            continue
        if m["half2"] and not 1 <= hour <= 12:
            continue
        if not m["half2"] and _DURATION_BEFORE.search(text[: m.start()]):
            continue
        minute = 30 if (m["half"] or m["half2"]) else _number(m["m"]) if m["m"] else 0
        if m["kem"]:
            hour, minute = hour - 1, 60 - _number(m["kem"])
        if minute > 59:
            continue
        found.append((m.start(), {"hour": hour, "minute": minute, "part": _part(text, m.start(), m.end())}))
    spans.extend(clock_spans)

    short_spans = []
    for m in _SHORT.finditer(text):
        if inside(m.start()):
            continue
        short_spans.append(m.span())
        hour = _number(m["h"])
        if 1 <= hour <= 12:
            found.append((m.start(), {"hour": hour, "minute": 0, "part": m["p"]}))
    spans.extend(short_spans)

    for m in _DIGITAL.finditer(text):
        if inside(m.start()):
            continue
        hour = int(m["h"])
        if hour <= 24:
            found.append((m.start(), {"hour": hour, "minute": int(m["m"]) if m["m"] else 0,
                                      "part": _part(text, m.start(), m.end(), before=False)}))
    return [entry for _, entry in sorted(found, key=lambda item: item[0])]


def _datetime_keys(call: dict) -> list[str]:
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return []
    return [key for key, value in arguments.items() if isinstance(value, str) and _DATETIME.match(value)]


def has_datetime(call: dict) -> bool:
    """True when any string argument of the call starts with YYYY-MM-DDTHH:MM."""
    return bool(_datetime_keys(call))


def _wall(value: str) -> datetime:
    return datetime.strptime(value[:16], "%Y-%m-%dT%H:%M")


def _target(clock: dict, old: datetime, now: datetime) -> datetime | None:
    hour, minute, part = clock["hour"], clock["minute"], clock["part"]
    if part is None:  # bare: the model wrote the evening and the morning reading is still ahead
        morning = old - timedelta(hours=12)
        if (1 <= hour <= 11 and old.date() == now.date() and old.hour >= 12
                and (old.hour - 12, old.minute) == (hour, minute) and morning > now):
            return morning
        return None
    if hour >= 13:
        target = hour % 24
    elif part == "sáng":
        target = hour if hour < 12 else None
    elif part == "trưa":
        target = hour if hour >= 10 else hour + 12
    elif part in ("chiều", "tối"):
        target = hour + 12 if hour < 12 else None
    else:  # đêm, khuya
        target = hour + 12 if 9 <= hour <= 11 else None
    if target is None:
        return None
    return old.replace(hour=target, minute=minute)


def fix_call_clock(call: dict, request: str, current_time: str) -> dict:
    """The call with the time of day the request states, or `call` itself when no rule applies."""
    if _OFFSET.search(unicodedata.normalize("NFC", request).lower()):
        return call
    found = clocks(request)
    keys = _datetime_keys(call)
    if len(found) != 1 or not 1 <= len(keys) <= 2:
        return call
    arguments = call["arguments"]
    try:  # a value the model wrote need not be a real date ("2026-11-31"): leave the call to validation
        old = min(_wall(arguments[key]) for key in keys)
        target = _target(found[0], old, _wall(current_time))
    except ValueError:
        return call
    if target is None or target == old:
        return call
    shift = target - old
    new_arguments = dict(arguments)
    for key in keys:
        value = arguments[key]
        new_arguments[key] = (_wall(value) + shift).strftime("%Y-%m-%dT%H:%M") + value[16:]
    return {**call, "arguments": new_arguments}


def fix_calls_clock(calls: list[dict], request: str, current_time: str) -> list[dict]:
    """fix_call_clock on the first call with a datetime argument; the same list object when nothing changes."""
    for i, call in enumerate(calls):
        if has_datetime(call):
            fixed = fix_call_clock(call, request, current_time)
            if fixed is call:
                return calls
            return calls[:i] + [fixed] + calls[i + 1:]
    return calls
