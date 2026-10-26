"""Score a run on train (the only split with labels) and write metrics.json + per_sample.jsonl next to it.

    python -m lifemate.evaluate runs/<run>            # split read from the run's meta.json
    python -m lifemate.evaluate path/to/submission.jsonl --split train

The gold state_hash comes from replaying label.tool_calls in a fresh ToolEnv (this matches label.final_state on
every train sample).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data import default_data_root, fuzzy_params, import_zaic_tools, load_catalog, load_samples
from .scoring import score_sample


def label_kind(gold_calls: list[dict]) -> str:
    if not gold_calls:
        return "empty"
    return "ask" if any(c["name"] == "sys.ask_missing_info" for c in gold_calls) else "act"


def history_bucket(n_turns: int) -> str:
    return "h0" if n_turns == 0 else ("h2-6" if n_turns <= 6 else "h8+")


def evaluate(pred_path: Path, split: str, data_root: Path) -> dict:
    zaic_tools = import_zaic_tools(data_root, split)
    samples = {s["id"]: s for s in load_samples(data_root, split)}
    if not all("label" in s for s in samples.values()):
        raise SystemExit(f"split {split!r} has no labels; only train can be scored locally")
    fuzzy = fuzzy_params(load_catalog(data_root, split))
    preds = [json.loads(line) for line in open(pred_path, encoding="utf-8") if line.strip()]
    dupes = len(preds) - len({p["id"] for p in preds})
    if dupes:
        raise SystemExit(f"{pred_path}: {dupes} duplicate ids")

    rows = []
    for pred in preds:
        sample = samples[pred["id"]]
        gold_calls = sample["label"]["tool_calls"]
        env = zaic_tools.ToolEnv(split, sample["id"], root=data_root)
        for call in gold_calls:
            env.call(call["name"], call["arguments"])
        r = score_sample(pred["tool_calls"], pred.get("state_hash"), gold_calls, env.get_state_hash(), fuzzy)
        r.update(id=pred["id"], kind=label_kind(gold_calls), hist=history_bucket(len(sample["history"])),
                 n_pred=len(pred["tool_calls"]), n_gold=len(gold_calls))
        rows.append(r)

    def agg(group: list[dict]) -> dict:
        n = len(group)
        return {"n": n, "score": round(100 * sum(r["score"] for r in group) / n, 2),
                "f1_no_hash": round(100 * sum(r["f1"] for r in group) / n, 2),
                "hash_acc": round(100 * sum(r["hash_ok"] for r in group) / n, 2)}

    groups = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
        groups[r["hist"]].append(r)
    metrics = {"split": split, "pred": str(pred_path), "n_in_split": len(samples), "overall": agg(rows),
               "by_group": {k: agg(v) for k, v in sorted(groups.items())}}
    out_dir = pred_path.parent
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "per_sample.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return metrics


def summary_row(run_dir: Path) -> str:
    """One Markdown row for RESULTS.md from a run folder (meta.json + metrics.json if scored)."""
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    mpath = run_dir / "metrics.json"
    m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    g = (m or {}).get("by_group", {})
    cell = lambda k: str(g[k]["score"]) if k in g else "-"  # noqa: E731
    commit = (meta.get("git") or {}).get("commit") or "?"
    return " | ".join([
        "", meta["finished"][:16].replace("T", " "), commit, meta["split"], str(meta["n"]), meta["agent"], meta["asr"],
        str(m["overall"]["score"]) if m else "(submit)", str(m["overall"]["f1_no_hash"]) if m else "-",
        cell("act"), cell("ask"), cell("empty"), str(meta["mean_seconds"]), str(meta["max_seconds"]), "",
    ]).strip()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", type=Path, help="run folder or submission.jsonl")
    ap.add_argument("--split", default=None, help="default: from the run's meta.json")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--row-only", action="store_true", help="print the RESULTS.md row of a run without scoring")
    args = ap.parse_args(argv)

    if args.row_only:
        print(summary_row(args.pred))
        return {}
    pred = args.pred / "submission.jsonl" if args.pred.is_dir() else args.pred
    split = args.split
    if split is None:
        meta = pred.parent / "meta.json"
        split = json.loads(meta.read_text(encoding="utf-8"))["split"] if meta.exists() else "train"
    metrics = evaluate(pred, split, args.data_root)

    o = metrics["overall"]
    print(f"{split}: {o['n']}/{metrics['n_in_split']} samples  score {o['score']}  F1(no hash) {o['f1_no_hash']}  hash acc {o['hash_acc']}")
    for k, v in metrics["by_group"].items():
        print(f"  {k:6s} n={v['n']:4d}  score {v['score']:6.2f}  F1 {v['f1_no_hash']:6.2f}  hash {v['hash_acc']:6.2f}")
    if (pred.parent / "meta.json").exists():
        print("RESULTS.md row:\n" + summary_row(pred.parent))
    return metrics


if __name__ == "__main__":
    main()
