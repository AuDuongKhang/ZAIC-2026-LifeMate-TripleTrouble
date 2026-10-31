"""Where an ASR cache is still wrong: error rate by word category, frequent confusions, final vs history clips, worst clips.

    python -m lifemate.asr_errors phoasr-small-ft-a [--split train] [--top 30] [--worst 20]

Words are compared after lifemate.asr.wer normalization (so WER here matches asr_report). Each gold word gets one
category from its original spelling in audio_text:
  number   digits or Vietnamese number words (giờ, tiền, số điện thoại ...)
  foreign  brands, acronyms and English words: Latin-only words with f/j/w/z, all-caps acronyms, words mixing letters
           and digits (SJC, VTV6, Netflix, wifi, RON95)
  name     other capitalized words that do not start a sentence (people, places, shops)
  other    everything else
Also writes asr_cache/<split>/<name>.errors.md.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from .asr.cached import cache_path, cache_root, read_cache
from .asr.wer import DIGITS, normalize
from .asr_report import references
from .data import default_data_root, load_samples

NUMBER_WORDS = set(DIGITS) | {"mười", "mươi", "trăm", "nghìn", "triệu", "tỷ", "lăm", "mốt", "tư", "rưỡi", "lẻ", "linh"}
AMBIGUOUS_NUMBERS = {"không", "một", "năm"}  # also "no", "a", "year": numbers only next to another number
CATEGORIES = ("number", "foreign", "name", "other")
_SENTENCE_END = re.compile(r"[.!?;:…]$")
_LATIN = re.compile(r"^[A-Za-z0-9+&'\-]+$")


def word_category(word: str, sentence_start: bool) -> str:
    core = word.strip(".,!?;:…\"'()[]")
    toks = normalize(core)
    if any(ch.isdigit() for ch in core) and not any(ch.isalpha() for ch in core):
        return "number"
    letters = [ch for ch in core if ch.isalpha()]
    if _LATIN.match(core) and letters and (
            re.search(r"[fjwzFJWZ]", core) or (len(letters) >= 2 and core.isupper()) or any(ch.isdigit() for ch in core)):
        return "foreign"
    if letters and core[0].isupper() and not sentence_start:  # "Ba", "Sáu" in a sentence are mostly names
        return "name"
    if toks and all(t in NUMBER_WORDS for t in toks):
        return "number"
    return "other"


def gold_tokens(ref: str) -> list[tuple[str, str]]:
    """Normalized gold tokens with the category of the original word each came from (same tokens as normalize(ref))."""
    words, start = [], True
    for word in ref.split():
        words.append((word, word_category(word, start), normalize(word)))
        start = bool(_SENTENCE_END.search(word))
    out = []
    for i, (word, cat, toks) in enumerate(words):
        if cat == "number" and toks and set(toks) <= AMBIGUOUS_NUMBERS:
            near = [words[k][1] for k in (i - 1, i + 1) if 0 <= k < len(words)]
            cat = "number" if "number" in near else "other"
        out += [(t, cat) for t in toks]
    return out


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, str | None]]:
    """Levenshtein alignment as (op, ref index, hyp token); op is ok, sub, del or ins."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def analyze(rows: dict[str, dict], refs: dict[str, str]) -> dict:
    by_cat = {c: Counter() for c in CATEGORIES}
    subs, dels, ins = Counter(), Counter(), Counter()
    per_clip = []
    for clip, row in rows.items():
        if clip not in refs:
            continue
        gold = gold_tokens(refs[clip])
        ref = [t for t, _ in gold]
        hyp = normalize(row["text"])
        errors = 0
        for t, cat in gold:
            by_cat[cat]["words"] += 1
        for op, ri, ht in align(ref, hyp):
            if op == "ins":
                ins[ht] += 1
                errors += 1
                continue
            tok, cat = gold[ri]
            if op == "sub":
                by_cat[cat]["errors"] += 1
                subs[(tok, ht)] += 1
                errors += 1
            elif op == "del":
                by_cat[cat]["errors"] += 1
                dels[tok] += 1
                errors += 1
        w = errors / len(ref) if ref else float(bool(hyp))
        per_clip.append({"audio": clip, "wer": w, "history": bool(re.search(r"_u\d+\.wav$", clip)),
                         "ref": refs[clip], "hyp": row["text"]})
    return {"by_cat": by_cat, "subs": subs, "dels": dels, "ins": ins, "clips": per_clip}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def render(name: str, res: dict, top: int, worst: int) -> str:
    clips = res["clips"]
    errs = sum(c["errors"] for c in res["by_cat"].values())
    lines = [f"# ASR errors: {name}", "",
             f"{len(clips)} clips, mean WER {_mean([c['wer'] for c in clips]):.4f}; "
             f"final requests {_mean([c['wer'] for c in clips if not c['history']]):.4f}, "
             f"history turns {_mean([c['wer'] for c in clips if c['history']]):.4f}; "
             f"{sum(res['ins'].values())} inserted words", "",
             "## By word category (substitutions + deletions of gold words)", "",
             "| category | gold words | errors | error rate | share of errors |", "|---|---|---|---|---|"]
    for cat in CATEGORIES:
        c = res["by_cat"][cat]
        lines.append(f"| {cat} | {c['words']} | {c['errors']} | {c['errors'] / max(c['words'], 1):.3f} | "
                     f"{c['errors'] / max(errs, 1):.1%} |")
    buckets = Counter("0" if c["wer"] == 0 else "<0.2" if c["wer"] < 0.2 else "<0.5" if c["wer"] < 0.5 else ">=0.5"
                      for c in clips)
    lines += ["", "## Clips by WER", "", " | ".join(f"{k}: {buckets[k]}" for k in ("0", "<0.2", "<0.5", ">=0.5")), "",
              f"## Top {top} substitutions (gold -> heard)", ""]
    lines += [f"- {n} × {g} -> {h}" for (g, h), n in res["subs"].most_common(top)]
    lines += ["", f"## Top {top} deleted gold words", "", ", ".join(f"{w} ({n})" for w, n in res["dels"].most_common(top)),
              "", f"## Top {top} inserted words", "", ", ".join(f"{w} ({n})" for w, n in res["ins"].most_common(top)),
              "", f"## {worst} worst clips", ""]
    for c in sorted(clips, key=lambda c: -c["wer"])[:worst]:
        lines += [f"- `{c['audio']}` WER {c['wer']:.2f}", f"  - gold: {c['ref']}", f"  - asr:  {c['hyp']}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", help="cache name under asr_cache/<split>/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--worst", type=int, default=20)
    args = ap.parse_args(argv)

    path = cache_path(args.split, args.cache)
    if not path.exists():
        raise SystemExit(f"{path} not found")
    res = analyze(read_cache(path), references(load_samples(args.data_root, args.split)))
    text = render(args.cache, res, args.top, args.worst)
    print(text, end="")
    out = cache_root() / args.split / f"{args.cache}.errors.md"
    out.write_text(text, encoding="utf-8")
    print(f"-> {out}")
    return res


if __name__ == "__main__":
    main()
