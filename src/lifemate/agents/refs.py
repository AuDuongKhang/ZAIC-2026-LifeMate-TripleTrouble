"""Which ids a conversation has actually shown: scalars in tool results, and x-ref arguments that never appeared."""
from __future__ import annotations


def scalar_strings(obj, out: set[str]) -> set[str]:
    """Every scalar inside obj, as a string: the ids and values the agent has actually seen."""
    if isinstance(obj, dict):
        for v in obj.values():
            scalar_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            scalar_strings(v, out)
    elif obj is not None:
        out.add(str(obj))
    return out


def unseen_refs(spec: dict | None, arguments: dict, seen: set[str]) -> list[str]:
    """'param=value' for each x-ref argument (an id into another app's data) whose value never appeared in a tool
    result. On train, 1277 of 1278 gold x-ref values appear in an earlier result, so these are invented ids."""
    props = ((spec or {}).get("parameters") or {}).get("properties", {})
    bad = []
    for name, value in arguments.items():
        prop = props.get(name, {})
        if not (prop.get("x-ref") or (prop.get("items") or {}).get("x-ref")):
            continue
        for v in value if isinstance(value, list) else [value]:
            if str(v) not in seen:
                bad.append(f"{name}={v!r}")
    return bad


PHONE_PARAMS = ("phone", "phone_number")
_DIGIT_WORDS = {"không": "0", "linh": "0", "một": "1", "mốt": "1", "hai": "2", "ba": "3", "bốn": "4", "tư": "4",
                "năm": "5", "lăm": "5", "sáu": "6", "bảy": "7", "bẩy": "7", "tám": "8", "chín": "9"}


def _phone_key(value) -> str:
    """Digits only, with a leading +84 / 84 country code turned into 0."""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return "0" + digits[2:] if digits.startswith("84") and len(digits) >= 11 else digits


def spoken_numbers(text: str) -> set[str]:
    """Digit strings written in a text: runs of digits (spaces, dots and dashes inside a run are ignored) and runs of
    Vietnamese digit words ("không chín tám bốn ..."), the way transcripts spell phone numbers out."""
    out: set[str] = set()
    run = ""
    for token in str(text).lower().replace(",", " ").split():
        word = token.strip(".!?;:()\"'")
        if word and all(ch.isdigit() or ch in ".-+" for ch in word) and any(ch.isdigit() for ch in word):
            run += "".join(ch for ch in word if ch.isdigit())
        elif word in _DIGIT_WORDS:
            run += _DIGIT_WORDS[word]
        else:
            if run:
                out.add(run)
            run = ""
    if run:
        out.add(run)
    return {_phone_key(r) for r in out}


def ungrounded_phones(arguments: dict, seen: set[str], texts: list[str]) -> list[str]:
    """'param=value' for each phone argument whose digits appear neither in a tool result nor in the request or a
    history turn: a number the model made up (e.g. 0912345678). Phone parameters carry no x-ref, so unseen_refs
    misses them."""
    bad = []
    for name in PHONE_PARAMS:
        value = arguments.get(name)
        if not isinstance(value, (str, int)) or not str(value).strip():
            continue
        key = _phone_key(value)
        known = {_phone_key(s) for s in seen if any(ch.isdigit() for ch in s)}
        for text in texts:
            known |= spoken_numbers(text)
        if key not in known:
            bad.append(f"{name}={value!r}")
    return bad
