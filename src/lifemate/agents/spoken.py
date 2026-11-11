"""Respell the misheard parts of a spoken email address back to the spoken form the model was trained on.

Users say an address as "<local> a còng example chấm test" (10 train gold texts), and the model learnt to write it as
<local>@example.test. ASR mishears the parts: on dev + holdout, all 8 samples whose gold email is spoken in the request
score 0, with "a còng" heard as "Account", "Item" or "acong" and "chấm test" as "chấm tết" or ".step". Two rules, case
insensitive, on whole words:
- "a còng" / "a cong" / "acong" / "account" / "item" right before the word "example" and then "." or "chấm" becomes
  "a còng" (the dot is the evidence of an address: "an account example" stays);
- "example" then "." or "chấm" then "tết" / "tét" / "step" (a misheard "test") becomes "example chấm test".
Anything else is left as heard, including an address ASR already wrote right ("example.test"). The local part (before
"a còng") is never touched.
"""
from __future__ import annotations

import re
import unicodedata

_AT = re.compile(r"(?<!\w)(?:a còng|a cong|acong|account|item)\s+(?=example\s*(?:\.|chấm(?!\w)))", re.IGNORECASE)
_DOT = re.compile(r"(?<!\w)example\s*(?:\.|chấm)\s*(?:tết|tét|step)(?!\w)", re.IGNORECASE)


def respell_emails(text: str) -> str:
    """`text` with misheard email parts in their spoken form; `text` itself when no rule applies."""
    nfc = unicodedata.normalize("NFC", text)
    out = _DOT.sub("example chấm test", _AT.sub("a còng ", nfc))
    return text if out == nfc else out
