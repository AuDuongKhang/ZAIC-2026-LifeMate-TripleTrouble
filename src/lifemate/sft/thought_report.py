"""Reports for the reasoning-lines stop points (docs/superpowers/specs/2026-11-05-reasoning-lines-design.md).

    python -m lifemate.sft.thought_report runs/<run> [--show 30]
    python -m lifemate.sft.thought_report --thoughts train_records/thoughts/pilot1.jsonl --read 50

The first form reads a run's trace.jsonl and counts the samples that ended on a sentence with no call after it (the
agent read the sentence as a plain answer and stopped), with the run's mean and max seconds; it says `over the 2%
limit` when more than 2% of the samples did. --show K prints K random traces for reading. The second form writes
`<file stem>.read.md` next to the thoughts file: one random step of each of N random samples, with the request, the
history, the earlier calls and results (the gold calls replayed in one fresh train ToolEnv), the next call and its
sentence, ready to be marked true and useful, or not. Development tooling only: no agent code imports this module.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..agents.ask_guard import fold
from ..agents.react import CLOSING
from ..data import default_data_root, import_zaic_tools, load_samples
from .thoughts import _dump, _history_text, load_thoughts, step_calls

SHARE_LIMIT = 2.0  # percent of samples; above it the opt-in retry from the spec needs its own plan and gate run
TIME_LIMIT_S = 30


def sentence_without_call(row: dict) -> bool:
    """`row` is one trace.jsonl line. True when the agent stopped on a plain answer whose last line is not a closing
    text: a sentence that was not followed by a call. A closing text, a refusal (even after its own sentence), an ask,
    a time or token stop, an empty answer and a sample that raised are all normal endings."""
    agent = row.get("agent")
    if not agent or agent.get("stop") != "no_call" or not agent.get("steps"):
        return False
    text = (agent["steps"][-1].get("output") or "").strip()
    if not text:
        return False
    return tuple(fold(text.splitlines()[-1])) not in CLOSING


def run_report(rows: list[dict]) -> dict:
    n = len(rows)
    seconds = [r.get("seconds") or 0.0 for r in rows]
    no_call = sum(sentence_without_call(r) for r in rows)
    return {"n": n, "no_call_sentences": no_call, "share": round(100 * no_call / n, 2) if n else 0.0,
            "mean_seconds": round(sum(seconds) / n, 3) if n else 0.0, "max_seconds": round(max(seconds, default=0.0), 3),
            "over_30s": sum(s > TIME_LIMIT_S for s in seconds)}


def read_sheet(rows: list[dict], samples: dict[str, dict], zaic_tools, data_root: Path, n: int, seed: int = 0) -> str:
    """Markdown with one random step of each of n random thoughts rows (the seed fixes the pick)."""
    rng = random.Random(seed)
    picked = rng.sample(sorted(rows, key=lambda r: r["id"]), min(n, len(rows)))
    parts = ["# Sentence reading sheet\n",
             "Mark each sentence: true and useful, or not. True means it states only what the request, the history "
             "and the earlier results show.\n"]
    for i, row in enumerate(picked, 1):
        sample = samples[row["id"]]
        calls = step_calls(sample)
        k = rng.randrange(len(row["steps"]))
        env = zaic_tools.ToolEnv("train", sample["id"], root=data_root)  # a fresh env: the earlier results only
        earlier = [f"- {c['name']} {_dump(c['arguments'])} -> {_dump(env.call(c['name'], c['arguments']))}" for c in calls[:k]]
        nxt = "none (out of scope: nothing is called)" if calls[k] is None else f"{calls[k]['name']} {_dump(calls[k]['arguments'])}"
        parts.append("\n".join([
            f"## {i}. {sample['id']} (step {k + 1} of {len(row['steps'])})", "",
            f"**Request:** {sample['audio_text']}", "",
            "**History:**", "```", _history_text(sample["history"]), "```", "",
            "**Earlier calls and results:**", "\n".join(earlier) or "(none)", "",
            f"**Next call:** {nxt}", "",
            f"**Sentence:** {row['steps'][k]['thought']}", "",
            "**Verdict:** [ ] true and useful   [ ] not", ""]))
    return "\n".join(parts)


def _show(rows: list[dict], k: int, seed: int) -> None:
    for row in random.Random(seed).sample(rows, min(k, len(rows))):
        agent = row.get("agent") or {}
        print(f"\n--- {row['id']}  {row.get('seconds')}s  stop={agent.get('stop')}  error={'yes' if row.get('error') else 'no'}")
        print(f"request: {row.get('request')}")
        for i, step in enumerate(agent.get("steps") or [], 1):
            print(f"step {i}: {step.get('output')}" + (f"   REJECTED {step['rejected']}" if step.get("rejected") else ""))
        for call in row.get("calls") or []:
            print("call: " + json.dumps(call, ensure_ascii=False)[:300])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path, help="runs/<run> folder with a trace.jsonl")
    ap.add_argument("--show", type=int, default=0, help="print K random traces of the run")
    ap.add_argument("--thoughts", type=Path, help="a thoughts file to build a reading sheet from")
    ap.add_argument("--read", type=int, help="sentences in the reading sheet (with --thoughts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)
    if (args.run is None) == (args.thoughts is None):
        ap.error("give either a run folder or --thoughts FILE --read N")
    if args.thoughts is not None and not args.read:
        ap.error("--thoughts needs --read N")
    if args.run is not None and args.read:
        ap.error("--read goes with --thoughts, not with a run folder")

    if args.run is not None:
        with open(args.run / "trace.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        report = run_report(rows)
        print(json.dumps(report, indent=2))
        if report["share"] > SHARE_LIMIT:
            print(f"sentence without call: {report['share']}% of samples, over the 2% limit: the opt-in retry from the "
                  "design needs its own short plan and its own gate run")
        if args.show:
            _show(rows, args.show, args.seed)
        return

    zaic_tools = import_zaic_tools(args.data_root, "train")
    samples = {s["id"]: s for s in load_samples(args.data_root, "train")}
    sheet = read_sheet(list(load_thoughts(args.thoughts).values()), samples, zaic_tools, args.data_root, args.read, args.seed)
    out = args.thoughts.with_name(f"{args.thoughts.stem}.read.md")
    out.write_text(sheet, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
