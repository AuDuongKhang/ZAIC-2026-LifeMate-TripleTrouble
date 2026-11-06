"""Where a train run on gold text loses points, ranked by request pattern, failure type and gold tool.

    python -m lifemate.loss_report runs/<gold dev run> runs/<gold holdout run> [--top 15]

Each run needs per_sample.jsonl (lifemate.evaluate) and submission.jsonl, and must use gold ASR, so that the losses are
the agent's and not the transcript's (--allow-asr overrides this). A sample's loss is 100 * (1 - score); a table row's
"points" is its share of the run's mean score lost, so the rows of an exclusive table add up to 100 minus the score.
Patterns are keyword matches on the gold request: a sample takes the first one that matches in PATTERNS. They are a
rough first cut for finding what to read, not a measurement of intent. Writes loss_report.md next to the first run.

"State right" only means the hash matches: reads, asks and free text do not change it, so such a sample can still have
done the wrong thing. Its table splits each sample's loss equally over the differences between its gold and predicted
calls (see state_right_issues), which is a way to rank them, not what fixing one difference alone would gain.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .asr_impact import call_change
from .data import SYS_TOOL, default_data_root, fuzzy_params, load_catalog, load_samples
from .scoring import FUZZY_THRESHOLD, _tokens, call_matches, token_f1

PATTERNS = (
    ("self-correction", r"à không|à thôi|ờ thôi|mà thôi|thôi khỏi|\blộn\b|\bnhầm\b"),
    ("condition", r"\b(nếu|mà)\b.*\bthì\b|còn không|không thì"),
    ("fallback", r"(không đủ|thiếu|hết chỗ|hết vé).{0,40}\bthì\b"),
    ("pick best", r"rẻ nhất|sớm nhất|muộn nhất|gần nhất|cuối cùng|đắt nhất|mạnh hơn|nhanh nhất"),
    ("arithmetic", r"một nửa|chia đôi|đúng (số|bằng|tiền)|cho tròn|tổng"),
    ("relative time", r"trước .{0,25}(tiếng|phút)|sớm hơn|muộn hơn|\bdời\b|\blùi\b"),
    ("refers back", r"\b(đó|đấy|ấy|kia|nớ|hồi nãy|lúc nãy|vừa rồi)\b"),
    ("several steps", r"\b(rồi|xong|với lại|sau đó)\b"),
)
OTHER = "plain"


def pattern(text: str) -> str:
    text = text.lower()
    return next((name for name, regex in PATTERNS if re.search(regex, text)), OTHER)


def failure(row: dict, gold_calls: list[dict], pred_calls: list[dict]) -> str:
    """Why a sample lost points. State right but calls off (hash_ok) is its own type: only the F1 suffers."""
    if row["score"] >= 1:
        return "no loss"
    if row.get("hash_ok"):
        return "state right, calls differ"
    if not gold_calls:
        return "out of scope, made calls"
    if not pred_calls:
        return "no calls"
    return call_change(gold_calls, pred_calls)


def gold_tool(gold_calls: list[dict]) -> str:
    """The last gold call's tool, or the tool an ask is about: what the request finally wants done."""
    if not gold_calls:
        return "(out of scope)"
    last = gold_calls[-1]
    if last["name"] == "sys.ask_missing_info":
        return f"ask: {last['arguments'].get('tool_name')}"
    return last["name"]


def _kind(call: dict, catalog: dict[str, dict]) -> str:
    return "ask" if call["name"] == SYS_TOOL else (catalog.get(call["name"]) or {}).get("kind", "write")


def _free_text_shape(pred, gold) -> str:
    """How a free-text argument missed the token-F1 threshold: too long, too short, or other words at similar length."""
    p, g = len(_tokens(pred)), len(_tokens(gold))
    return "longer than gold" if p > 1.5 * g else "shorter than gold" if g > 1.5 * p else "other words"


def _pair_issue(pred: dict, gold: dict, kind: str, fuzzy: set[str]) -> str:
    """Why two calls to the same tool do not match."""
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if kind == "ask":
        return "ask about other params"
    wrong = [k for k in pa.keys() | ga.keys() if k not in pa or k not in ga or
             (token_f1(pa[k], ga[k]) <= FUZZY_THRESHOLD if k in fuzzy else pa[k] != ga[k])]
    if wrong and all(k in fuzzy and k in pa and k in ga for k in wrong):
        k = max(wrong, key=lambda key: abs(len(_tokens(pa[key])) - len(_tokens(ga[key]))))
        return f"free text: {_free_text_shape(pa[k], ga[k])}"
    if kind == "read":
        return "read args: extra key" if set(pa) > set(ga) else "read args: missing key" if set(pa) < set(ga) else "read args: value"
    return "write args, no state effect"


def state_right_issues(gold_calls: list[dict], pred_calls: list[dict], catalog: dict[str, dict],
                       fuzzy_by_tool: dict[str, set[str]]) -> list[str]:
    """One label per difference: exact matches go first, then same-tool pairs, then unmatched reads of the same app
    count as the wrong read tool; what is left is missing (gold only) or extra (predicted only)."""
    gold, pred = list(gold_calls), list(pred_calls)
    for g in list(gold):
        hit = next((p for p in pred if call_matches(p, g, fuzzy_by_tool.get(g["name"], set()))), None)
        if hit is not None:
            gold.remove(g)
            pred.remove(hit)
    issues = []
    for g in list(gold):
        hit = next((p for p in pred if p["name"] == g["name"]), None)
        if hit is not None:
            issues.append(_pair_issue(hit, g, _kind(g, catalog), fuzzy_by_tool.get(g["name"], set())))
            gold.remove(g)
            pred.remove(hit)
    for g in list(gold):
        if _kind(g, catalog) != "read":
            continue
        app = g["name"].split(".", 1)[0]
        hit = next((p for p in pred if _kind(p, catalog) == "read" and p["name"].split(".", 1)[0] == app), None)
        if hit is not None:
            issues.append("other read tool, same app")
            gold.remove(g)
            pred.remove(hit)
    return issues + [f"missing {_kind(g, catalog)}" for g in gold] + [f"extra {_kind(p, catalog)}" for p in pred]


def analyze(rows: dict[str, dict], preds: dict[str, dict], samples: dict[str, dict],
            catalog: dict[str, dict] | None = None) -> list[dict]:
    catalog = catalog or {}
    fuzzy = fuzzy_params(catalog)
    out = []
    for i in sorted(rows.keys() & preds.keys() & samples.keys()):
        s, r = samples[i], rows[i]
        gold = s["label"]["tool_calls"]
        out.append({"id": i, "loss": round(100 * (1 - r["score"]), 2), "kind": r.get("kind"),
                    "pattern": pattern(s["audio_text"]), "failure": failure(r, gold, preds[i].get("tool_calls") or []),
                    "gold_tool": gold_tool(gold), "history": bool(s["history"]), "text": s["audio_text"],
                    "issues": state_right_issues(gold, preds[i].get("tool_calls") or [], catalog, fuzzy)
                    if r.get("hash_ok") and r["score"] < 1 else []})
    return out


def issue_table(rows: list[dict]) -> list[str]:
    """Points of the state-right samples, split equally over each sample's differences."""
    n = len(rows) or 1
    points, seen = defaultdict(float), defaultdict(set)
    for r in rows:
        for issue in r["issues"]:
            points[issue] += r["loss"] / len(r["issues"])
            seen[issue].add(r["id"])
    lines = ["| difference | samples | points |", "|---|---|---|"]
    lines += [f"| {k} | {len(seen[k])} | {points[k] / n:.2f} |" for k in sorted(points, key=lambda k: -points[k])]
    return lines


def table(rows: list[dict], key: str, top: int | None = None, total: int | None = None) -> list[str]:
    """Rows of one Markdown table: groups by total points lost, biggest first. total is the run's sample count, for
    tables of a subset of it."""
    n = total or len(rows) or 1
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    ranked = sorted(groups.items(), key=lambda kv: -sum(r["loss"] for r in kv[1]))
    lines = [f"| {key} | samples | points | mean loss | samples losing |", "|---|---|---|---|---|"]
    for name, g in ranked[:top]:
        lost = sum(r["loss"] for r in g)
        lines.append(f"| {name} | {len(g)} | {lost / n:.2f} | {lost / len(g):.1f} | {sum(r['loss'] > 0 for r in g)} |")
    return lines


def render(name: str, rows: list[dict], top: int) -> str:
    n = len(rows) or 1
    score = 100 - sum(r["loss"] for r in rows) / n
    lines = [f"## {name}", "", f"{len(rows)} samples, score {score:.2f}, {100 - score:.2f} points lost.", ""]
    for title, key, limit in (("Request pattern (keyword match)", "pattern", None), ("Failure type", "failure", None),
                              (f"Gold tool (top {top})", "gold_tool", top)):
        lines += [f"### {title}", "", *table(rows, key, limit), ""]
    lines += ["### State right: what differs in the calls", "", *issue_table(rows), ""]
    lines += ["### Pattern and failure, biggest first", ""]
    pairs = [{**r, "pair": f"{r['pattern']} / {r['failure']}"} for r in rows if r["loss"] > 0]
    lines += [*table(pairs, "pair", top, len(rows)), ""]
    return "\n".join(lines)


def read_jsonl(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in map(json.loads, f) if row}


def load_run(run: Path, samples: dict[str, dict], allow_asr: bool, catalog: dict[str, dict] | None = None) -> list[dict]:
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("split", "train") != "train":
        raise SystemExit(f"{run}: loss_report needs labels, so it only works on train runs")
    if not allow_asr and meta.get("asr") != "gold":
        raise SystemExit(f"{run}: asr is {meta.get('asr')!r}, not gold; losses would mix in ASR errors (--allow-asr)")
    if not (run / "per_sample.jsonl").exists():
        raise SystemExit(f"{run}: no per_sample.jsonl; run lifemate.evaluate first")
    return analyze(read_jsonl(run / "per_sample.jsonl"), read_jsonl(run / "submission.jsonl"), samples, catalog)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--allow-asr", action="store_true")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)

    samples = {s["id"]: s for s in load_samples(args.data_root, "train")}
    catalog = load_catalog(args.data_root, "train")
    parts = ["# Loss report", ""]
    for run in args.runs:
        parts.append(render(run.name, load_run(run, samples, args.allow_asr, catalog), args.top))
    text = "\n".join(parts)
    print(text)
    out = args.runs[0] / "loss_report.md"
    out.write_text(text, encoding="utf-8")
    print(f"-> {out}")
    return out


if __name__ == "__main__":
    main()
