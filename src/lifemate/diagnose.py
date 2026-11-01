"""Where a train run loses points: which gold calls were never made, which were made with wrong arguments, and why the
agent stopped. Reads submission.jsonl (+ trace.jsonl when present); needs no ToolEnv.

    python -m lifemate.diagnose runs/<run>       # prints the breakdown and writes diagnose.json next to it

Counted over the gold calls of act/ask samples. Gold calls are paired with predicted calls greedily: exact matches
first (scoring.call_matches), then, per gold call left, the first unused predicted call with the same name.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import default_data_root, fuzzy_params, load_catalog, load_samples
from .evaluate import label_kind
from .scoring import FUZZY_THRESHOLD, call_matches, token_f1


def _same_value(key: str, pred, gold, fuzzy: set[str]) -> bool:
    return token_f1(pred, gold) > FUZZY_THRESHOLD if key in fuzzy else pred == gold


def diagnose(run_dir: Path, data_root: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")).get("split") != "train":
        raise SystemExit(f"{run_dir}: diagnose needs labels, so it only works on train runs")
    samples = {s["id"]: s for s in load_samples(data_root, "train")}
    catalog = load_catalog(data_root, "train")
    fuzzy = fuzzy_params(catalog)
    preds = [json.loads(line) for line in open(run_dir / "submission.jsonl", encoding="utf-8") if line.strip()]
    traces = {}
    if (run_dir / "trace.jsonl").exists():
        for line in open(run_dir / "trace.jsonl", encoding="utf-8"):
            if line.strip():
                t = json.loads(line)
                traces[t["id"]] = t

    out = {"samples": 0, "gold_calls": 0, "matched": 0, "missing_read": 0, "missing_write": 0, "missing_ask": 0, "calls_with_extra_key": 0,
           "extra_pred_calls": 0, "fewer_calls": 0, "more_calls": 0, "out_of_scope_with_calls": 0}
    extra_key, missing_key, wrong_value, stop = Counter(), Counter(), Counter(), Counter()
    for pred in preds:
        gold_calls = samples[pred["id"]]["label"]["tool_calls"]
        pred_calls = pred["tool_calls"]
        if (t := traces.get(pred["id"])) and t.get("agent"):
            stop[t["agent"]["stop"]] += 1
        if label_kind(gold_calls) == "empty":
            out["out_of_scope_with_calls"] += bool(pred_calls)
            continue
        out["samples"] += 1
        out["gold_calls"] += len(gold_calls)
        out["fewer_calls"] += len(pred_calls) < len(gold_calls)
        out["more_calls"] += len(pred_calls) > len(gold_calls)
        free_g, free_p = list(range(len(gold_calls))), list(range(len(pred_calls)))
        for j in list(free_g):
            g = gold_calls[j]
            hit = next((i for i in free_p if call_matches(pred_calls[i], g, fuzzy.get(g["name"], set()))), None)
            if hit is not None:
                out["matched"] += 1
                free_g.remove(j)
                free_p.remove(hit)
        for j in free_g:
            g = gold_calls[j]
            same = next((i for i in free_p if pred_calls[i]["name"] == g["name"]), None)
            if same is None:
                out["missing_" + {"read": "read", "write": "write", "meta": "ask"}[catalog[g["name"]]["kind"]]] += 1
                continue
            free_p.remove(same)
            pa, ga = pred_calls[same].get("arguments") or {}, g["arguments"]
            out["calls_with_extra_key"] += bool(pa.keys() - ga.keys())
            for key in pa.keys() - ga.keys():
                extra_key[f"{g['name']}.{key}"] += 1
            for key in ga.keys() - pa.keys():
                missing_key[f"{g['name']}.{key}"] += 1
            for key in pa.keys() & ga.keys():
                if not _same_value(key, pa[key], ga[key], fuzzy.get(g["name"], set())):
                    wrong_value[f"{g['name']}.{key}"] += 1
        gold_names = {g["name"] for g in gold_calls}
        out["extra_pred_calls"] += sum(pred_calls[i]["name"] not in gold_names for i in free_p)
    for name, counter in (("extra_key", extra_key), ("missing_key", missing_key), ("wrong_value", wrong_value),
                          ("stop", stop)):
        out[name] = dict(counter.most_common())
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)
    result = diagnose(args.run, args.data_root)
    (args.run / "diagnose.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    scalars = {k: v for k, v in result.items() if not isinstance(v, dict)}
    print(json.dumps(scalars, ensure_ascii=False))
    for name in ("stop", "extra_key", "missing_key", "wrong_value"):
        counter = result[name]
        print(f"{name} (summed over keys: {sum(counter.values())}):", dict(list(counter.items())[:15]))
    return result


if __name__ == "__main__":
    main()
