"""Score an offline rewrite of a recorded train run's calls without a GPU.

    python -m lifemate.sim <rewrite> runs/<run> [--kw key=value ...]

A rewrite is a function (trace line, train catalog) -> new call list or None. The trace line is one line of the run's
trace.jsonl ({"id", "request", "calls": [{"name", "arguments", "result"}], ...}), the catalog maps tool name to spec.
Returning None leaves the sample unchanged (it keeps its recorded submission line); returning a list of
{"name", "arguments"} replays those calls in a fresh train ToolEnv for the exact tool_calls and state hash. Writes
runs/<run>/sim/<name>/ (submission, per_sample, metrics) and prints the compare line against the run itself.
The simulation assumes the agent would have made the same calls up to the point where the rewrite changes them.
Named rewrites are factories in REWRITES; --kw key=value (JSON values) are passed to the factory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from .compare import compare, load_scores, verdict
from .data import default_data_root, import_zaic_tools, load_catalog
from .evaluate import evaluate
from .utils import parse_kw

Rewrite = Callable[[dict, "dict[str, dict]"], "list[dict] | None"]

REWRITES: dict[str, Callable[..., Rewrite]] = {}


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
    label = "_".join([args.rewrite] + [f"{k}{v}" for k, v in kw.items()])
    r = rewrite_run(args.run, args.data_root, label, REWRITES[args.rewrite](**kw))
    print(f"{label}: {format_result(r)}", flush=True)
    return r


if __name__ == "__main__":
    main()
