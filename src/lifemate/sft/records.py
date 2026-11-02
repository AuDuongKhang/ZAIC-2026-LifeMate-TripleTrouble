"""Write SFT training records: one conversation per train sample, rendered by replaying its gold calls through the agent.

    python -m lifemate.sft.records --name gold-pool --subset pool        # -> train_records/gold-pool.jsonl
    python -m lifemate.sft.records --name gold-dev --subset dev          # eval-loss records only, never trained on

Only the pool is trained on in experiments (dev and holdout stay unseen); `--subset all` is for the final retrain.
Few-shot examples (--examples, default 4 like the best Phase 1 agent) come from the same bank the agent uses at
inference: the pool for pool/dev/holdout records, every train sample for `all`. A sample is never its own example.
Replays that do not reproduce exactly the gold calls are dropped and counted, never written with partial calls.
`--mask K` adds K function-masked copies after each record (lifemate.sft.mask), seeded by --seed and the sample id.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ..agents.fewshot import ExampleBank
from ..agents.prompts import DEFAULT_PROMPT, PROMPTS
from ..asr.base import GoldTranscriptASR
from ..data import REPO_ROOT, default_data_root, import_zaic_tools, load_catalog, load_samples
from ..evaluate import label_kind
from ..infer import build_task
from ..splits import load_splits
from .mask import mask_record
from .replay import replay
from .tokenize import render

RECORDS_DIR = REPO_ROOT / "train_records"
SUBSETS = ("pool", "dev", "holdout", "all")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", choices=SUBSETS, required=True)
    ap.add_argument("--text", default="gold", help="gold transcripts (cache:<asr_name> comes in a later task)")
    ap.add_argument("--mask", type=int, default=0, help="function-masked copies per record, after the original")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS))
    ap.add_argument("--examples", type=int, default=4, help="few-shot examples in the system prompt (0 = none)")
    ap.add_argument("--limit", type=int, default=None, help="first N samples of the subset only")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507", help="for the token report; 'none' skips it")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--out-dir", type=Path, default=RECORDS_DIR)
    args = ap.parse_args(argv)
    if args.text != "gold":
        ap.error("--text cache:<name> is not implemented yet; use --text gold")

    zaic_tools = import_zaic_tools(args.data_root, "train")
    samples = load_samples(args.data_root, "train")
    catalog = load_catalog(args.data_root, "train")
    splits = load_splits()
    wanted = {s["id"] for s in samples} if args.subset == "all" else set(splits[args.subset])
    chosen = [s for s in samples if s["id"] in wanted][: args.limit]
    bank = None
    if args.examples:
        bank_ids = {s["id"] for s in samples} if args.subset == "all" else set(splits["pool"])
        bank = ExampleBank(samples, bank_ids, catalog)

    asr = GoldTranscriptASR()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.jsonl"
    written, dropped = Counter(), []
    with open(out, "w", encoding="utf-8") as f:
        for sample in chosen:
            task = build_task(sample, asr, catalog, args.data_root, "train")
            env = zaic_tools.ToolEnv("train", sample["id"], root=args.data_root)  # fresh env per record
            rec = replay(sample, task, env, args.prompt, args.examples, bank)
            if rec is None:
                dropped.append(sample["id"])
                continue
            rec["source"] = "gold"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for k in range(args.mask):
                masked = mask_record(rec, random.Random(f"{args.seed}:{sample['id']}:{k}"))
                f.write(json.dumps(masked, ensure_ascii=False) + "\n")
            written[label_kind(sample["label"]["tool_calls"])] += 1

    print(f"-> {out}: {sum(written.values())} records {dict(written)}"
          + (f" + {args.mask} masked copies each" if args.mask else "") + f", {len(dropped)} replays dropped"
          + (f" {dropped[:20]}" if dropped else ""), flush=True)
    if args.tokenizer != "none":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        with open(out, encoding="utf-8") as f:
            lengths = [len(render(tok, r["messages"], r["tools"])) for r in map(json.loads, f)]
        if lengths:
            print(f"tokens per record: mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}, "
                  f"total {sum(lengths)}", flush=True)
    return out


if __name__ == "__main__":
    main()
