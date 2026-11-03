"""Weekday table for the system prompt: resolves "thứ ba tuần sau" and the like by lookup instead of calendar maths.
Weeks start on Monday, which is how train gold resolves them (Thu 2026-10-01 + "thứ ba tuần sau" = 2026-10-06)."""
from __future__ import annotations

from datetime import date, timedelta

WEEKDAYS = ("thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật")  # Monday first
RELATIVE_DAYS = {0: "hôm nay", 1: "ngày mai", 2: "ngày mốt"}


def date_table(current_time: str) -> str:
    """One line per day from today to the Sunday of next week, e.g. "2026-10-06 thứ ba (tuần sau)"."""
    today = date.fromisoformat(current_time[:10])  # the local date as written, no timezone conversion
    monday = today - timedelta(days=today.weekday())
    lines = ["Calendar (weeks start on Monday):"]
    for offset in range((monday + timedelta(days=13) - today).days + 1):
        day = today + timedelta(days=offset)
        labels = [RELATIVE_DAYS[offset]] if offset in RELATIVE_DAYS else []
        labels.append("tuần này" if day < monday + timedelta(days=7) else "tuần sau")
        lines.append(f"{day.isoformat()} {WEEKDAYS[day.weekday()]} ({', '.join(labels)})")
    return "\n".join(lines)
