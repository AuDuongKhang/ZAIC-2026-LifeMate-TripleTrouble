"""Score the ask guard (lifemate.agents.ask_guard) on a recorded train run without a GPU.

    python -m lifemate.guard_sim runs/<run>                                  # min_score 0.5, margin 0.2
    python -m lifemate.guard_sim runs/<run> --min-score 0.3 --margin 0.1
    python -m lifemate.guard_sim runs/<run> --grid                           # 3 x 3 settings

Walks each sample's recorded calls in order, as the agent would. At the first write the guard stops, the call list
becomes "the calls before it, then the ask" (the agent ends its turn after an ask), and that list is replayed in a
fresh train ToolEnv for the exact tool_calls and state hash. Samples where the guard never fires keep their recorded
submission line. Writes runs/<run>/guard_sim/<setting>/ (submission, per_sample, metrics) and prints the compare
line against the run itself. The simulation assumes the agent would have made the same calls up to that point,
which holds because the guard only acts before a write is sent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents.ask_guard import ask_instead, candidate_lists
from .compare import compare, load_scores, verdict
from .data import default_data_root, import_zaic_tools, load_catalog
from .evaluate import evaluate

GRID = [(s, m) for s in (0.3, 0.5, 0.7) for m in (0.1, 0.2, 0.3)]


def guarded_calls(request: str, calls: list[dict], catalog: dict[str, dict], min_score: float,
                  margin: float) -> list[dict] | None:
    """The call list with the guard applied, or None when it never fires."""
    candidates: dict[str, list[dict]] = {}
    for i, call in enumerate(calls):
        ask = ask_instead(request, call, catalog.get(call["name"]), candidates, min_score, margin)
        if ask is not None:
            return [{"name": c["name"], "arguments": c["arguments"]} for c in calls[:i]] + [ask]
        candidate_lists(call.get("result"), candidates)
    return None


def simulate(run: Path, data_root: Path, min_score: float, margin: float) -> dict:
    zaic_tools, catalog = import_zaic_tools(data_root, "train"), load_catalog(data_root, "train")
    recorded = {json.loads(line)["id"]: line for line in open(run / "submission.jsonl", encoding="utf-8") if line.strip()}
    out = run / "guard_sim" / f"s{min_score}_m{margin}"
    out.mkdir(parents=True, exist_ok=True)
    fired = 0
    with open(out / "submission.jsonl", "w", encoding="utf-8") as f:
        for line in open(run / "trace.jsonl", encoding="utf-8"):
            trace = json.loads(line)
            calls = guarded_calls(trace["request"], trace["calls"], catalog, min_score, margin)
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


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="scored train run folder (trace.jsonl, submission.jsonl, per_sample.jsonl)")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--grid", action="store_true", help=f"try {len(GRID)} settings")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)
    results = []
    for s, m in GRID if args.grid else [(args.min_score, args.margin)]:
        r = simulate(args.run, args.data_root, s, m)
        results.append(r)
        print(f"min_score {s} margin {m}: fired {r['fired']}  a {r['mean_a']}  b {r['mean_b']}  diff {r['diff']:+} "
              f"[95% CI {r['ci_low']:+}, {r['ci_high']:+}]  better {r['better']} / worse {r['worse']}  {verdict(r)}",
              flush=True)
    return results


if __name__ == "__main__":
    main()
