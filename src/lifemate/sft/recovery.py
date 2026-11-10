"""Recovery records: training prompts for the state after the model's own wrong read.

Gold replays only show on-path states. After one wrong read (a guessed filter, a name cut short) the model is somewhere it
never trained on. From a scored run of a model that makes such mistakes, `recovery_sample` keeps the leading read calls of
a trace that differ from the gold call at their position as `prefix_calls`. `lifemate.sft.records --extra-samples` then
renders the sample with `replay`: the LLM emits the prefix, then the gold calls the prefix did not already make, in the
train environment the sample points at, and the final state must equal the gold replay's. The record carries
`context_turns`, the prefix length, and `sft.tokenize.encode_record` leaves those turns unlabelled, so the loss never
teaches the mistake itself, only what to do after it.

    python -m lifemate.sft.recovery --run runs/<run> --name <n> --max 200 [--hf-repo <dataset>]
                                                                          # -> train_records/synth/recovery-<n>.jsonl

The run must be a scored pool run of the untrained model (`react-fewshot-k4`, gold text, `--subset pool`): sft-v2 has seen
the pool and makes too few mistakes on it. Development only; nothing at inference imports this.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from ..data import REPO_ROOT, default_data_root, import_zaic_tools, load_catalog, load_samples

OUT_DIR = REPO_ROOT / "train_records" / "synth"


def _key(call: dict) -> str:
    return json.dumps([call["name"], call["arguments"]], sort_keys=True, ensure_ascii=False)


def recovery_sample(sample: dict, trace: dict, catalog: dict[str, dict], n: int) -> dict | None:
    """A copy of `sample` with env_id, id "<id>~r<n>" and prefix_calls, the longest leading run of read calls of the
    trace, when at least one of them differs from the gold call at its position. None when there is no such prefix (no
    leading read, or every read is the gold one). The run ends at the first write, ask or unknown tool."""
    prefix = []
    for call in trace["calls"]:
        if (catalog.get(call["name"]) or {}).get("kind") != "read":
            break
        prefix.append({"name": call["name"], "arguments": call["arguments"]})
    gold = sample["label"]["tool_calls"]
    if not prefix or all(i < len(gold) and _key(c) == _key(gold[i]) for i, c in enumerate(prefix)):
        return None
    new = copy.deepcopy(sample)
    new.update(id=f"{sample['id']}~r{n}", env_id=sample["id"], prefix_calls=copy.deepcopy(prefix), source="recovery")
    return new


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="scored train run folder with trace.jsonl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--hf-repo", default=None, help="private HF dataset to upload the file and report to (synth/<file>)")
    ap.add_argument("--max", type=int, default=200, help="most records to write")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    from ..infer import build_task
    from ..asr.base import GoldTranscriptASR
    from ..splits import load_splits
    from .replay import replay
    from .synth import uses_holdout_app

    samples = {s["id"]: s for s in load_samples(args.data_root, "train")}
    catalog = load_catalog(args.data_root, "train")
    splits = load_splits()
    pool, held_out = set(splits["pool"]), set(splits["holdout_apps"])
    traces = [json.loads(line) for line in open(args.run / "trace.jsonl", encoding="utf-8") if line.strip()]
    outside = sorted(t["id"] for t in traces if t["id"] not in pool)
    if outside:
        ap.error(f"{len(outside)} traces are not pool samples (for example {outside[:3]}): recovery records come from a "
                 "pool run only, so dev and holdout stay unseen")
    zaic_tools = import_zaic_tools(args.data_root, "train")
    asr = GoldTranscriptASR()
    rows, candidates, why = [], 0, Counter()
    for trace in traces:
        base = samples[trace["id"]]
        sample = recovery_sample(base, trace, catalog, 0)
        if sample is None:
            why["on path or no leading read"] += 1
            continue
        candidates += 1
        if uses_holdout_app(sample["prefix_calls"], held_out):  # pool samples have holdout apps installed too
            why["prefix calls a holdout app"] += 1
            continue
        task = build_task(sample, asr, catalog, args.data_root, "train")
        env = zaic_tools.ToolEnv("train", base["id"], root=args.data_root)
        record = replay(sample, task, env, fresh_env=lambda base=base: zaic_tools.ToolEnv("train", base["id"], root=args.data_root))
        if record is None:
            why["replay mismatch or changed state"] += 1
            continue
        rows.append(sample)
        if len(rows) >= args.max:
            break
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"recovery-{args.name}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = {"traces": len(traces), "candidates": candidates, "usable": len(rows), "not_usable_by_reason": dict(why),
              "prefix_length": dict(Counter(len(r["prefix_calls"]) for r in rows))}
    (args.out_dir / f"recovery-{args.name}.report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"-> {out}: {report}", flush=True)
    if len(rows) < 100:
        print(f"WARNING: only {len(rows)} usable records (the plan stops below 100): report to K before training", flush=True)
    if args.hf_repo:  # the training notebook reads it from the dataset: another session has not got this folder
        from huggingface_hub import HfApi

        for path in (out, args.out_dir / f"recovery-{args.name}.report.json"):
            HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"synth/{path.name}", repo_id=args.hf_repo,
                                repo_type="dataset")
        print(f"uploaded synth/{out.name} to {args.hf_repo}", flush=True)
    return out


if __name__ == "__main__":
    main()
