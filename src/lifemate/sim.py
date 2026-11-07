"""Score an offline rewrite of a recorded train run's calls without a GPU.

    python -m lifemate.sim <rewrite> runs/<run> [--kw key=value ...]

A rewrite is a function (trace line, train catalog) -> new call list or None. The trace line is one line of the run's
trace.jsonl ({"id", "request", "calls": [{"name", "arguments", "result"}], ...}), the catalog maps tool name to spec.
Returning None leaves the sample unchanged (it keeps its recorded submission line); returning a list of
{"name", "arguments"} replays those calls in a fresh train ToolEnv for the exact tool_calls and state hash. Writes
runs/<run>/sim/<name>/ (submission, per_sample, metrics) and prints the compare line against the run itself.
The simulation assumes the agent would have made the same calls up to the point where the rewrite changes them.
Named rewrites are factories in REWRITES; --kw key=value (JSON values) are passed to the factory.
run_vote takes the other runs as --kw others='["runs/a","runs/b"]'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from .agents.date_fix import fix_call_dates
from .agents.freetext import freetext_rewrite
from .agents.minimal_reads import minimal_reads_rewrite
from .agents.react import decides, vote_key
from .compare import compare, load_scores, verdict
from .data import default_data_root, import_zaic_tools, load_catalog, load_samples
from .evaluate import evaluate
from .utils import parse_kw

Rewrite = Callable[[dict, "dict[str, dict]"], "list[dict] | None"]


def run_vote(others: list[str]) -> Rewrite:
    """Majority over the write and ask calls of the run being rewritten and the runs in `others` (folders with a
    trace.jsonl). Each run's key per sample is the vote keys of its write and ask calls, in order (free text and ids
    of reads do not count, as in the prompt vote). Only a run that makes at least one write or ask votes, and a run
    without the sample does not. The key with most votes wins when it has at least 2; ties and no majority keep the
    first run (the one being rewritten). The winner's whole call list replaces the first run's. Returns None when the
    first run already is in the majority. This estimates the room for a vote; a real one has to work on proposals before
    they are sent, because every call on a ToolEnv is scored."""
    traces = []
    for folder in others:
        with open(Path(folder) / "trace.jsonl", encoding="utf-8") as f:
            traces.append({t["id"]: t for t in map(json.loads, filter(str.strip, f))})

    def rewrite(trace: dict, catalog: dict[str, dict]) -> list[dict] | None:
        votes: dict[tuple, list[dict]] = {}
        for t in [trace] + [run.get(trace["id"]) for run in traces]:
            calls = (t or {}).get("calls") or []
            if not decides(calls, catalog):
                continue
            key = tuple(vote_key(c, catalog.get(c["name"])) for c in calls if decides([c], catalog))
            votes.setdefault(key, []).append(t)
        top = max((len(v) for v in votes.values()), default=0)
        winners = [v for v in votes.values() if len(v) == top]
        if top < 2 or len(winners) != 1 or winners[0][0] is trace:
            return None
        return [{"name": c["name"], "arguments": c["arguments"]} for c in winners[0][0]["calls"]]
    return rewrite


def date_fix(times: dict[str, str] | None = None, data_root: Path | None = None) -> Rewrite:
    """Put the date of the request's only date phrase into each call's single date argument (agents/date_fix.py). The
    trace has no current_time, so it comes from the train samples (`times` {id: current_time} overrides, for tests)."""
    if times is None:
        times = {s["id"]: s["context"]["current_time"] for s in load_samples(data_root or default_data_root(), "train")}

    def rewrite(trace: dict, catalog: dict[str, dict]) -> list[dict] | None:
        now = times.get(trace["id"])
        if now is None:
            return None
        fixed = [fix_call_dates(c, catalog.get(c["name"]), trace.get("request", ""), now) for c in trace["calls"]]
        if all(f is c for f, c in zip(fixed, trace["calls"])):
            return None
        return [{"name": c["name"], "arguments": c["arguments"]} for c in fixed]
    return rewrite


REWRITES: dict[str, Callable[..., Rewrite]] = {"freetext": freetext_rewrite, "minimal_reads": minimal_reads_rewrite,
                                               "run_vote": run_vote,
                                               "date_fix": date_fix}


def rewrite_run(run: Path, data_root: Path, name: str, rewrite: Rewrite, out_dir: Path | None = None) -> dict:
    """Apply `rewrite` to every trace line of `run`, score the result, and compare it with the run.

    Output goes to <out_dir or run/"sim">/<name>/. Returns compare(...) of run vs rewritten, plus `fired` (samples the
    rewrite changed) and `out` (the output folder)."""
    zaic_tools, catalog = import_zaic_tools(data_root, "train"), load_catalog(data_root, "train")
    recorded = {json.loads(line)["id"]: line for line in open(run / "submission.jsonl", encoding="utf-8") if line.strip()}
    out = (out_dir or run / "sim") / name
    out.mkdir(parents=True, exist_ok=True)
    fired = 0
    with open(out / "submission.jsonl", "w", encoding="utf-8") as f:
        for line in open(run / "trace.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            trace = json.loads(line)
            calls = rewrite(trace, catalog)
            if calls is None:
                f.write(recorded[trace["id"]].rstrip("\n") + "\n")
                continue
            fired += 1
            env = zaic_tools.ToolEnv("train", trace["id"], root=data_root)
            for c in calls:
                env.call(c["name"], c["arguments"])
            f.write(json.dumps({"id": trace["id"], "tool_calls": env.get_tool_calls(),
                                "state_hash": env.get_state_hash()}, ensure_ascii=False) + "\n")
    evaluate(out / "submission.jsonl", "train", data_root)
    return dict(compare(load_scores(run), load_scores(out)), fired=fired, out=out)


def format_result(r: dict) -> str:
    return (f"fired {r['fired']}  a {r['mean_a']}  b {r['mean_b']}  diff {r['diff']:+} "
            f"[95% CI {r['ci_low']:+}, {r['ci_high']:+}]  better {r['better']} / worse {r['worse']}  {verdict(r)}")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rewrite", help="registered rewrite name (sim.REWRITES)")
    ap.add_argument("run", type=Path, help="scored train run folder (trace.jsonl, submission.jsonl, per_sample.jsonl)")
    ap.add_argument("--kw", nargs="*", default=[], metavar="KEY=VALUE", help="factory arguments (JSON values)")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)
    if args.rewrite not in REWRITES:
        raise SystemExit(f"unknown rewrite {args.rewrite!r}; known: {', '.join(sorted(REWRITES)) or 'none'}")
    kw = parse_kw(args.kw)
    label = "_".join([args.rewrite] + [f"{k}{len(v) if isinstance(v, (list, dict)) else v}" for k, v in kw.items()])
    r = rewrite_run(args.run, args.data_root, label, REWRITES[args.rewrite](**kw))
    print(f"{label}: {format_result(r)}", flush=True)
    return r


if __name__ == "__main__":
    main()
