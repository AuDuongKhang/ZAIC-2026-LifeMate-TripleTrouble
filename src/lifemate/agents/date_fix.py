"""Correct the date of a call when the request holds exactly one relative date phrase.

The model sometimes writes the wrong calendar day for "thứ ba tuần sau" and the like. resolve_date turns the phrase into an
ISO date by calendar maths (weeks start on Monday, which is how train gold resolves them), and fix_call_dates puts that date
into the call's single date-valued string argument, keeping any time part. It changes an argument of a call the model sent
and never sends a new call. No schema or app name decides anything except the description of the argument (a past range
is left alone). Shared by the simulator (`lifemate.sim`, rewrite "date_fix") and the agent (`date_fix=True`).
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
_PAST_RANGE = re.compile(r"tu thoi diem|qua khu|da qua|truoc day|lich su|since")  # matched on folded words


def resolve_date(request: str, current_time: str) -> str | None:
    """The ISO date of the request's only date phrase, or None (no phrase, several, or a day that cannot be told).

    hôm nay, ngày mai, ngày mốt, sáng/trưa/chiều/tối mai; thứ X tuần sau (that weekday of next week); thứ X này (this
    week, None when already past); thứ X without a qualifier and chủ nhật (the next one, today included)."""
    matches = list(_PHRASE.finditer(unicodedata.normalize("NFC", request).lower()))
    if len(matches) != 1:
        return None
    m = matches[0]
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


def fix_call_dates(call: dict, spec: dict | None, request: str, current_time: str) -> dict:
    """A copy of `call` whose single date-valued string argument carries the date of the request's phrase, or `call`
    itself when nothing applies: no phrase, no such argument, two of them, or an argument that names a past range."""
    dated = [k for k, v in call.get("arguments", {}).items() if isinstance(v, str) and _DATE_VALUE.match(v)]
    if len(dated) != 1 or _names_past_range(spec, dated[0]):
        return call
    new = resolve_date(request, current_time)
    value = call["arguments"][dated[0]]
    if new is None or value[:10] == new:
        return call
    return dict(call, arguments=dict(call["arguments"], **{dated[0]: new + value[10:]}))
