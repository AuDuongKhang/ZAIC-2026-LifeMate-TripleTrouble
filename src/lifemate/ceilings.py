"""Score ceilings per error bucket: what a train run would score if one kind of error were replaced by gold.

    python -m lifemate.ceilings runs/<run>

Writes runs/<run>/ceilings/<bucket>/ (submission, per_sample, metrics, through lifemate.sim) and runs/<run>/ceilings.md.
Each bucket is a ceiling for that kind of error, not a forecast: buckets overlap, so they do not add up, and the
simulation assumes the agent would have made the same calls otherwise (see lifemate.sim). Uses train labels, so it
refuses any other split.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .agents.ask_guard import SYS_TOOL
from .data import default_data_root, load_samples
from .sim import Rewrite, format_result, rewrite_run

_DATE_VALUE = re.compile(r"^\d{4}-\d\d-\d\d|^\d\d:\d\d")
_ALL_ARGS = frozenset({"free", "id", "date", "num", "enum_bool", "string"})
_READ_MODES = {"drop_extra", "add_missing", "fix_args", "all"}
_ASK_MODES = {"gold_asks", "agent_asks", "all"}

BUCKETS: dict[str, dict] = {
    "free_text": {"arg_types": {"free"}},
    "id": {"arg_types": {"id"}},
    "date": {"arg_types": {"date"}},
    "num": {"arg_types": {"num"}},
    "string": {"arg_types": {"string"}},
    "enum_bool": {"arg_types": {"enum_bool"}},
    "all_args": {"arg_types": _ALL_ARGS},
    "reads": {"reads": "all"},
    "reads_drop_extra": {"reads": "drop_extra"},
    "reads_add_missing": {"reads": "add_missing"},
    "reads_fix_args": {"reads": "fix_args"},
    "ask_act": {"ask": "all"},
    "ask_gold_asks": {"ask": "gold_asks"},
    "ask_agent_asks": {"ask": "agent_asks"},
    "write_set": {"write_set": True},
    "out_of_scope": {"out_of_scope": True},
    "all_args+reads": {"arg_types": _ALL_ARGS, "reads": "all"},
    "all_args+reads+ask": {"arg_types": _ALL_ARGS, "reads": "all", "ask": "all"},
    "everything": {"arg_types": _ALL_ARGS, "reads": "all", "ask": "all", "write_set": True, "out_of_scope": True},
}


def arg_type(spec: dict | None, key: str, value) -> str:
    """One of free, id, date, num, enum_bool, string. `spec` is the tool's catalog spec, None for an unknown tool."""
    prop = ((spec or {}).get("parameters") or {}).get("properties", {}).get(key, {})
    if prop.get("x-match"):
        return "free"
    if prop.get("x-ref") or (prop.get("items") or {}).get("x-ref"):
        return "id"
    if (isinstance(value, str) and _DATE_VALUE.match(value)) or prop.get("format") in ("date", "date-time", "time"):
        return "date"
    if isinstance(value, bool) or prop.get("type") == "boolean" or prop.get("enum"):
        return "enum_bool"
    if isinstance(value, (int, float)) or prop.get("type") in ("integer", "number"):
        return "num"
    return "string"


def _plain(calls: list[dict]) -> list[dict]:
    return [{"name": c["name"], "arguments": dict(c["arguments"])} for c in calls]


def _kind(catalog: dict[str, dict], call: dict) -> str | None:
    return (catalog.get(call["name"]) or {}).get("kind")


def _align(pred: list[dict], gold: list[dict]) -> dict[int, int]:
    """{pred index: gold index}, same tool name only. Each pred call in turn takes its best remaining gold call."""
    pairs: dict[int, int] = {}
    for i, c in enumerate(pred):
        left = [j for j, g in enumerate(gold) if g["name"] == c["name"] and j not in pairs.values()]
        if left:
            pairs[i] = max(left, key=lambda j: sum(gold[j]["arguments"].get(k) == v for k, v in c["arguments"].items()))
    return pairs


def _fix_args(calls: list[dict], gold: list[dict], catalog: dict[str, dict], types) -> list[dict]:
    pairs, out = _align(calls, gold), []
    for i, c in enumerate(calls):
        if i in pairs and _kind(catalog, c) == "write":
            want, args = gold[pairs[i]]["arguments"], dict(c["arguments"])
            for key in dict.fromkeys([*args, *want]):  # a dict, so argument order never depends on the hash seed
                if arg_type(catalog.get(c["name"]), key, want.get(key, args.get(key))) in types:
                    if key in want:
                        args[key] = want[key]
                    else:
                        del args[key]
            c = {"name": c["name"], "arguments": args}
        out.append(c)
    return out


def _fix_reads(calls: list[dict], gold: list[dict], catalog: dict[str, dict], mode: str) -> list[dict]:
    """Reads go first: they do not change the state, and the score does not depend on call order."""
    pred = [c for c in calls if _kind(catalog, c) == "read"]
    rest = [c for c in calls if _kind(catalog, c) != "read"]
    want = [c for c in gold if _kind(catalog, c) == "read"]
    pairs = _align(pred, want)
    if mode == "all":
        new = want
    elif mode == "drop_extra":
        new = [c for i, c in enumerate(pred) if i in pairs]
    elif mode == "add_missing":
        new = pred + [g for j, g in enumerate(want) if j not in pairs.values()]
    else:
        new = [dict(c, arguments=want[pairs[i]]["arguments"]) if i in pairs else c for i, c in enumerate(pred)]
    return _plain(new) + rest


def _fix_write_set(calls: list[dict], gold: list[dict], catalog: dict[str, dict]) -> list[dict]:
    pairs = _align(calls, gold)
    keep = [c for i, c in enumerate(calls) if _kind(catalog, c) != "write" or i in pairs]
    return keep + _plain([g for j, g in enumerate(gold) if _kind(catalog, g) == "write" and j not in pairs.values()])


def _same_calls(a: list[dict], b: list[dict]) -> bool:
    key = lambda c: json.dumps(c, ensure_ascii=False, sort_keys=True)  # noqa: E731
    return sorted(map(key, a)) == sorted(map(key, b))


def fix_rewrite(samples: dict[str, dict], arg_types: set[str] = frozenset(), reads: str | None = None,
                ask: str | None = None, write_set: bool = False, out_of_scope: bool = False) -> Rewrite:
    """A lifemate.sim rewrite that replaces the chosen kinds of error with gold. Returns None when nothing changes.

    arg_types: the arg_type values to fix, on write calls that align with a gold call. reads: drop_extra, add_missing,
    fix_args or all. ask: replace the whole sample with gold when the agent and gold disagree on asking (gold_asks: gold
    asks and the agent does not; agent_asks: the reverse; all: both). write_set: add missing writes, drop extra ones.
    out_of_scope: a gold [] sample with calls becomes an empty call list; only `ask` and this flag touch those samples.
    """
    if reads is not None and reads not in _READ_MODES:
        raise ValueError(f"unknown reads mode {reads!r}; known: {sorted(_READ_MODES)}")
    if ask is not None and ask not in _ASK_MODES:
        raise ValueError(f"unknown ask mode {ask!r}; known: {sorted(_ASK_MODES)}")

    def rewrite(trace: dict, catalog: dict[str, dict]) -> list[dict] | None:
        gold, pred = _plain(samples[trace["id"]]["label"]["tool_calls"]), _plain(trace["calls"])
        gold_asks, agent_asks = any(c["name"] == SYS_TOOL for c in gold), any(c["name"] == SYS_TOOL for c in pred)
        if (ask in ("gold_asks", "all") and gold_asks and not agent_asks) or (
                ask in ("agent_asks", "all") and agent_asks and not gold_asks):
            return gold
        if not gold:
            return [] if out_of_scope and pred else None
        out = pred
        if arg_types:
            out = _fix_args(out, gold, catalog, arg_types)
        if reads:
            out = _fix_reads(out, gold, catalog, reads)
        if write_set:
            out = _fix_write_set(out, gold, catalog)
        return None if _same_calls(out, pred) else out

    return rewrite


def main(argv: list[str] | None = None) -> dict[str, dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="scored train run folder (trace.jsonl, submission.jsonl, per_sample.jsonl)")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)
    meta = json.loads((args.run / "meta.json").read_text(encoding="utf-8"))
    if meta.get("split") != "train":
        raise SystemExit(f"{args.run}: split is {meta.get('split')!r}, ceilings need train labels")
    samples = {s["id"]: s for s in load_samples(args.data_root, "train")}
    results, rows = {}, []
    for bucket, kw in BUCKETS.items():
        r = rewrite_run(args.run, args.data_root, bucket, fix_rewrite(samples, **kw), args.run / "ceilings")
        print(f"{bucket}: {format_result(r)}", flush=True)
        results[bucket] = r
        rows.append(f"| {bucket} | {r['fired']} | {r['mean_b']} | {r['diff']:+} | [{r['ci_low']:+}, {r['ci_high']:+}] "
                    f"| {r['better']} | {r['worse']} |")
    base = next(iter(results.values()))["mean_a"]
    (args.run / "ceilings.md").write_text(
        f"# Score ceilings: {args.run.name}\n\nRun score {base}. Each bucket replaced by gold, alone; they overlap, so "
        "they do not add up.\n\n| Bucket | Fired | Score | Diff | 95% CI | Better | Worse |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
