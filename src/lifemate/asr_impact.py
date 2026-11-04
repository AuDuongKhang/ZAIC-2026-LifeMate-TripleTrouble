"""How much of the score ASR costs, and which ASR errors cost it: two train runs of the same agent, one on gold text
and one on ASR text, joined per sample with the WER of the transcript the agent saw.

    python -m lifemate.asr_impact runs/<gold run> runs/<asr run> [--cache pho-large-ft-b] [--top 25]

Both runs need per_sample.jsonl (lifemate.evaluate) and submission.jsonl; the ASR run needs trace.jsonl (its `request`
is the transcript of the final turn). `--cache` also scores the history turns, which the trace does not keep.
A sample's loss is 100 * (gold-run score - ASR-run score). Each sample falls in one cause bucket:
  exact      transcripts are exact, so any loss is not ASR (the floor of run-to-run noise)
  history    final request exact, a history turn misheard
  final:<c>  final request misheard; <c> is the rarest-first category of a wrong gold word
             (name, foreign, number, other; see lifemate.asr_errors)
and writes asr_impact.md and asr_impact.jsonl next to the ASR run.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .asr.cached import cache_path, read_cache
from .asr.wer import normalize, wer
from .asr_errors import align, gold_tokens
from .data import default_data_root, load_samples

PRIORITY = ("name", "foreign", "number", "other")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def error_categories(ref: str, hyp: str) -> set[str]:
    """Categories of the gold words the hypothesis got wrong (substituted or deleted)."""
    gold = gold_tokens(ref)
    cats = set()
    for op, ri, _ in align([t for t, _ in gold], normalize(hyp)):
        if op in ("sub", "del"):
            cats.add(gold[ri][1])
    return cats


def value_kind(key: str, value) -> str:
    if key == "id" or key.endswith("_id") or key.endswith("_ids"):
        return "id"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str) and (_DATE.match(value) or re.match(r"^\d{1,2}:\d{2}", value)):
        return "date/time"
    return "text/other"


def call_change(gold_calls: list[dict], asr_calls: list[dict]) -> str:
    """What differs between the two runs' calls: same, ask flip, other tools, or which kind of argument."""
    if gold_calls == asr_calls:
        return "same calls"
    asks = lambda calls: any(c["name"] == "sys.ask_missing_info" for c in calls)  # noqa: E731
    if asks(gold_calls) != asks(asr_calls):
        return "ask/act flip"
    names = lambda calls: [c["name"] for c in calls]  # noqa: E731
    if names(gold_calls) != names(asr_calls):
        return "other tools"
    kinds = set()
    for g, a in zip(gold_calls, asr_calls):
        for k in g["arguments"].keys() | a["arguments"].keys():
            if g["arguments"].get(k) != a["arguments"].get(k):
                kinds.add(value_kind(k, g["arguments"].get(k, a["arguments"].get(k))))
    return "args: " + "+".join(sorted(kinds)) if kinds else "same calls"


def analyze(samples: dict, gold_rows: dict, asr_rows: dict, gold_preds: dict, asr_preds: dict, requests: dict,
            history_wer: dict[str, float] | None = None) -> list[dict]:
    """One row per sample id present in every input."""
    ids = sorted(set(samples) & set(gold_rows) & set(asr_rows) & set(gold_preds) & set(asr_preds) & set(requests))
    out = []
    for i in ids:
        ref, hyp = samples[i]["audio_text"], requests[i]
        w = wer(ref, hyp)
        hw = (history_wer or {}).get(i)
        if w == 0:
            cause = "history" if hw else "exact"
        else:
            cats = error_categories(ref, hyp)
            cause = "final:" + next((c for c in PRIORITY if c in cats), "other")
        out.append({"id": i, "kind": gold_rows[i]["kind"], "loss": round(100 * (gold_rows[i]["score"] - asr_rows[i]["score"]), 2),
                    "final_wer": round(w, 3), "history_wer": None if hw is None else round(hw, 3), "cause": cause,
                    "change": call_change(gold_preds[i]["tool_calls"], asr_preds[i]["tool_calls"]),
                    "gold_text": ref, "asr_text": hyp})
    return out


def _bucket(w: float) -> str:
    return "0" if w == 0 else "(0,0.1]" if w <= 0.1 else "(0.1,0.3]" if w <= 0.3 else ">0.3"


def render(rows: list[dict], top: int) -> str:
    n = len(rows)
    total = sum(r["loss"] for r in rows)
    lost = [r for r in rows if r["loss"] > 0]
    won = [r for r in rows if r["loss"] < 0]
    lines = [f"# ASR impact", "", f"{n} samples. Mean score loss {total / n:+.2f} points (gold run minus ASR run): "
             f"{len(lost)} samples lose, {len(won)} gain, {n - len(lost) - len(won)} unchanged. "
             f"Losses add up to {sum(r['loss'] for r in lost) / n:.2f} points, gains give back "
             f"{-sum(r['loss'] for r in won) / n:.2f}.", ""]

    def table(title: str, groups: dict[str, list[dict]], order=None):
        lines.extend([f"## {title}", "", "| group | samples | mean loss | share of net loss | losing samples |", "|---|---|---|---|---|"])
        for k in order or sorted(groups, key=lambda g: -sum(r["loss"] for r in groups[g])):
            g = groups.get(k, [])
            if g:
                s = sum(r["loss"] for r in g)
                lines.append(f"| {k} | {len(g)} | {s / len(g):+.2f} | {s / total:.0%} | {sum(r['loss'] > 0 for r in g)} |"
                             if total else f"| {k} | {len(g)} | {s / len(g):+.2f} | - | {sum(r['loss'] > 0 for r in g)} |")
        lines.append("")

    by = defaultdict(list)
    for r in rows:
        by[_bucket(r["final_wer"])].append(r)
    table("By WER of the final request", by, ["0", "(0,0.1]", "(0.1,0.3]", ">0.3"])
    by = defaultdict(list)
    for r in rows:
        by[r["cause"]].append(r)
    table("By cause (rarest wrong word category first: name, foreign, number, other)", by)
    by = defaultdict(list)
    for r in lost:
        by[r["change"]].append(r)
    table("What changed in the calls (losing samples only)", by)
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    table("By sample kind", by)

    lines.extend([f"## {top} biggest losses", ""])
    for r in sorted(lost, key=lambda r: -r["loss"])[:top]:
        lines += [f"- `{r['id']}` loss {r['loss']:.0f}, final WER {r['final_wer']}, {r['cause']}, {r['change']}",
                  f"  - gold: {r['gold_text']}", f"  - asr:  {r['asr_text']}"]
    return "\n".join(lines) + "\n"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold_run", type=Path)
    ap.add_argument("asr_run", type=Path)
    ap.add_argument("--cache", default=None, help="ASR cache name; adds the history turns' WER")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)
    samples = {s["id"]: s for s in load_samples(args.data_root, "train")}
    by_id = lambda rows: {r["id"]: r for r in rows}  # noqa: E731
    gold_rows = by_id(_jsonl(args.gold_run / "per_sample.jsonl"))
    asr_rows = by_id(_jsonl(args.asr_run / "per_sample.jsonl"))
    gold_preds = by_id(_jsonl(args.gold_run / "submission.jsonl"))
    asr_preds = by_id(_jsonl(args.asr_run / "submission.jsonl"))
    requests = {t["id"]: t["request"] for t in _jsonl(args.asr_run / "trace.jsonl") if t.get("request") is not None}
    history_wer = None
    if args.cache:
        cache = read_cache(cache_path("train", args.cache))
        history_wer = {}
        for i, s in samples.items():
            ws = [wer(t["audio_text"], cache[t["audio"]]["text"]) for t in s["history"]
                  if t["role"] == "user" and t.get("audio") in cache]
            if ws:
                history_wer[i] = sum(ws) / len(ws)
    rows = analyze(samples, gold_rows, asr_rows, gold_preds, asr_preds, requests, history_wer)
    if not rows:
        raise SystemExit("the runs share no sample ids")
    text = render(rows, args.top)
    (args.asr_run / "asr_impact.md").write_text(text, encoding="utf-8")
    (args.asr_run / "asr_impact.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(text)
    return rows


if __name__ == "__main__":
    main()
