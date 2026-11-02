"""Word and character error rate of ASR transcripts against train audio_text.

Normalization: NFC, lowercase, punctuation and symbols stripped, standalone digit tokens spelled out in Vietnamese
("5 giờ" == "năm giờ"; digit strings with a leading 0, like phone numbers, digit by digit), and spelling variants
unified ("ngàn" -> "nghìn", "linh" -> "lẻ"). Digits inside words ("4g", "vtv3", "ron95") are kept. Old and new tone
placement are the same word ("hòa" == "hoà", "hủy" == "huỷ"), and colloquial tens read as written digits
("chín lăm" == "95" == "chín mươi lăm").
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

DIGITS = "không một hai ba bốn năm sáu bảy tám chín".split()
SCALES = ["", "nghìn", "triệu", "tỷ"]
VARIANTS = {"ngàn": "nghìn", "linh": "lẻ"}
_DIGIT_TOKEN = re.compile(r"[0-9]+")
_TONE_MARKS = "̣̀́̃̉"
_OLD_TONE = re.compile(rf"(o|u)([{_TONE_MARKS}])(a|e|y)(?![a-zđ])")  # NFD, lowercase, open syllable only
_TENS = set(DIGITS[2:])


def _group_words(n: int, full: bool) -> list[str]:
    """Words for 0 < n < 1000; `full` reads a zero hundreds digit ("không trăm"), as inside a larger number."""
    h, t, u = n // 100, n // 10 % 10, n % 10
    words = [DIGITS[h], "trăm"] if full or h else []
    if t == 0:
        if u and words:
            words.append("lẻ")
    elif t == 1:
        words.append("mười")
    else:
        words += [DIGITS[t], "mươi"]
    if u:
        words.append("mốt" if u == 1 and t >= 2 else "lăm" if u == 5 and t >= 1 else DIGITS[u])
    return words


def number_words(n: int) -> str:
    """Vietnamese reading of 0 <= n < 10**12."""
    if not 0 <= n < 10**12:
        raise ValueError(f"out of range: {n}")
    if n == 0:
        return DIGITS[0]
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    words: list[str] = []
    for scale in range(len(groups) - 1, -1, -1):
        if groups[scale]:
            words += _group_words(groups[scale], full=bool(words)) + ([SCALES[scale]] if scale else [])
    return " ".join(words)


def _spell(token: str) -> list[str]:
    if not _DIGIT_TOKEN.fullmatch(token):
        return [VARIANTS.get(token, token)]
    if (len(token) > 1 and token[0] == "0") or len(token) > 12:
        return [DIGITS[int(d)] for d in token]
    return number_words(int(token)).split()


def _tone_style(text: str) -> str:
    """Old-style tone placement ("hòa", "khóa", "hủy") -> new style ("hoà", "khoá", "huỷ"), as both appear in the data."""
    return unicodedata.normalize("NFC", _OLD_TONE.sub(r"\1\3\2", unicodedata.normalize("NFD", text)))


def _spoken_tens(words: list[str]) -> list[str]:
    """Colloquial "chín lăm", "hai mốt" -> "chín mươi lăm", "hai mươi mốt", as digits are spelled ("95" -> ...)."""
    out = []
    for i, w in enumerate(words):
        out.append(w)
        if (w in _TENS and i + 1 < len(words) and words[i + 1] in ("lăm", "mốt")
                and (i == 0 or words[i - 1] not in ("mươi", "mười"))):
            out.append("mươi")
    return out


def normalize(text: str) -> list[str]:
    text = _tone_style(unicodedata.normalize("NFC", text).lower())
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return _spoken_tens([w for token in text.split() for w in _spell(token)])


def _edit_distance(r: Sequence, h: Sequence) -> int:
    prev = list(range(len(h) + 1))  # Levenshtein, one row at a time
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r, h = " ".join(normalize(ref)), " ".join(normalize(hyp))
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def mean_wer(traces: list[dict], samples: dict[str, dict]) -> float | None:
    """Mean final-turn WER over trace rows that have a transcript; None when there are none."""
    scores = [wer(samples[t["id"]]["audio_text"], t["request"]) for t in traces if t.get("request") is not None]
    return round(sum(scores) / len(scores), 4) if scores else None
