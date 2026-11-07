"""Write SFT training records: one conversation per train sample, rendered by replaying its gold calls through the agent.

    python -m lifemate.sft.records --name gold-pool --subset pool        # -> train_records/gold-pool.jsonl
    python -m lifemate.sft.records --name gold-dev --subset dev          # eval-loss records only, never trained on

Only the pool is trained on in experiments (dev and holdout stay unseen); `--subset all` is for the final retrain.
Few-shot examples (--examples, default 4 like the best Phase 1 agent) come from the same bank the agent uses at
inference: the pool for pool/dev/holdout records, every train sample for `all`. A sample is never its own example.
Replays that do not reproduce exactly the gold calls are dropped and counted, never written with partial calls.
`--text cache:<asr_name>` takes the user turns from an ASR transcript cache (asr_cache/train/<asr_name>.jsonl) instead
of the gold transcripts; the calls and results stay the gold ones. Repeat --text to write one record per source and
sample in the same file. `--mask K` adds K function-masked copies (lifemate.sft.mask) after the record of the first
--text source, seeded by --seed and the sample id; `--perturb` also shuffles and thins their schemas
(lifemate.sft.perturb):

    python -m lifemate.sft.records --name v2-pool --subset pool --text gold --text cache:pho-large-fp16 --mask 1

`--remix FILE` (lifemate.sft.remix) adds one schema-paraphrased copy of each record after its masked ones, when the
table rewrites at least one app the record shows: new tool and app names, new descriptions, parameter synonyms; holdout
apps in the table are refused. Without it the records are unchanged.

`--extra-samples FILE` (repeatable) adds synthetic samples (lifemate.sft.synth) after the subset's own: rendered
like real ones, with the same --mask/--perturb copies, but only with gold text (they have no audio, so `cache:`
sources skip them). Each runs in the train environment named by its `env_id`, which must belong to the pool unless
--subset is `all`; they are never few-shot examples.

`--thoughts FILE` (lifemate.sft.thoughts) renders one reasoning sentence before each tool call, and before the refusal
of an out-of-scope sample, in every copy of a sample (gold, masked and ASR). It needs `--prompt v3` and cannot be
combined with `--extra-samples`. A sample with no row in FILE is dropped with all its copies and counted; more than
MAX_MISSING of the chosen samples without a row stops the run before anything is written. Rows for samples outside
the chosen subset are ignored, and a row whose call names differ from the gold calls raises ValueError.

`--leave-app-out P` (a share between 0 and 1) renders that share of the records with the apps of their own gold calls
(the asked tool's app for an ask) left out of the few-shot examples, chosen per (sample, source) from --seed, so the
model also trains on prompts where no example shows its app, as for an app it never saw. Every record then carries a
`leave_app_out` boolean (masked and perturbed copies inherit it); without the option the records are unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter
from pathlib import Path

from ..agents.fewshot import ExampleBank
from ..agents.prompts import DEFAULT_PROMPT, PROMPTS
from ..asr.base import ASR, GoldTranscriptASR
from ..asr.cached import CachedASR
from ..data import REPO_ROOT, default_data_root, import_zaic_tools, load_catalog, load_samples
from ..evaluate import label_kind
from ..infer import build_task
from ..splits import load_splits
from .mask import mask_record
from .perturb import perturb_record
from .remix import applies, remix_record
from .replay import replay
from .synth import env_id, uses_holdout_app
from .thoughts import load_thoughts
from .tokenize import render

RECORDS_DIR = REPO_ROOT / "train_records"
SUBSETS = ("pool", "dev", "holdout", "all")
MAX_MISSING = 0.05  # --thoughts: share of chosen samples that may lack a row before the run stops


def _catalog_apps(data_root: Path) -> list[dict]:
    return json.loads((data_root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))["apps"]


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", choices=SUBSETS, required=True)
    ap.add_argument("--text", action="append", default=None,
                    help="gold (default) or cache:<asr_name>; repeat for one record per source")
    ap.add_argument("--mask", type=int, default=0, help="function-masked copies of the first --text source's record")
    ap.add_argument("--perturb", action="store_true", help="perturb the schemas of masked copies (lifemate.sft.perturb)")
    ap.add_argument("--extra-samples", action="append", type=Path, default=[], metavar="FILE",
                    help="synthetic samples (jsonl) rendered after the subset; repeatable")
    ap.add_argument("--thoughts", type=Path, default=None, metavar="FILE",
                    help="reasoning sentences (lifemate.sft.thoughts); needs --prompt v3")
    ap.add_argument("--remix", type=Path, default=None, metavar="FILE",
                    help="schema paraphrase table (lifemate.sft.remix): one remixed copy after the masked ones")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS))
    ap.add_argument("--examples", type=int, default=4, help="few-shot examples in the system prompt (0 = none)")
    ap.add_argument("--date-hints", action="store_true", help="add the weekday table to the system prompt")
    ap.add_argument("--leave-app-out", type=float, default=None, metavar="P",
                    help="share in [0, 1] of the records whose few-shot examples leave out the apps of their own gold calls")
    ap.add_argument("--limit", type=int, default=None, help="first N samples of the subset only")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507", help="for the token report; 'none' skips it")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--out-dir", type=Path, default=RECORDS_DIR)
    args = ap.parse_args(argv)
    if args.leave_app_out is not None and not 0 <= args.leave_app_out <= 1:
        ap.error("--leave-app-out is a share between 0 and 1")
    sources: list[tuple[ASR, str]] = []
    for text in args.text or ["gold"]:
        if text == "gold":
            sources.append((GoldTranscriptASR(), "gold"))
        elif text.startswith("cache:") and text[len("cache:"):]:
            sources.append((CachedASR(text[len("cache:"):], split="train"), "asr"))
        else:
            ap.error(f"--text {text!r}: use gold or cache:<asr_name>")

    if args.thoughts and args.prompt != "v3":
        ap.error("--thoughts needs --prompt v3 (the rule that asks for a sentence before each call)")
    if args.thoughts and args.extra_samples:
        ap.error("--thoughts cannot be combined with --extra-samples (synthetic samples have no sentences)")

    remix_table = None
    if args.remix:
        from .remix import check_table

        try:
            remix_table = json.loads(args.remix.read_text(encoding="utf-8"))
            check_table(remix_table, load_splits()["holdout_apps"], known={a["id"] for a in _catalog_apps(args.data_root)})
        except (OSError, ValueError) as e:
            ap.error(f"--remix {args.remix}: {e}")
    zaic_tools = import_zaic_tools(args.data_root, "train")
    samples = load_samples(args.data_root, "train")
    catalog = load_catalog(args.data_root, "train")
    splits = load_splits()
    wanted = {s["id"] for s in samples} if args.subset == "all" else set(splits[args.subset])
    chosen = [s for s in samples if s["id"] in wanted][: args.limit]
    thoughts, missing = {}, []
    if args.thoughts:
        thoughts = load_thoughts(args.thoughts)
        missing = [s["id"] for s in chosen if s["id"] not in thoughts]
        if len(missing) > MAX_MISSING * len(chosen):
            ap.error(f"{len(missing)} of {len(chosen)} samples have no sentences in {args.thoughts} "
                     f"(limit {MAX_MISSING:.0%}): {missing[:10]}")
        chosen = [s for s in chosen if s["id"] in thoughts]
    extras = []
    for path in args.extra_samples:
        with open(path, encoding="utf-8") as f:
            extras += [json.loads(line) for line in f if line.strip()]
    allowed = {s["id"] for s in samples} if args.subset == "all" else set(splits["pool"])
    held_out = set(splits["holdout_apps"])
    for extra in extras:
        if args.subset != "all" and uses_holdout_app(extra["label"]["tool_calls"] + extra.get("prefix_calls", []), held_out):
            ap.error(f"extra sample {extra['id']} calls a holdout app: holdout apps stay out of experiment training data")
        if env_id(extra) not in allowed:
            ap.error(f"extra sample {extra['id']} uses a non-pool environment {env_id(extra)} (only --subset all "
                     "may use every train environment)")
    bank = None
    if args.examples:
        bank_ids = {s["id"] for s in samples} if args.subset == "all" else set(splits["pool"])
        bank = ExampleBank(samples, bank_ids, catalog)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.jsonl"
    kinds, written, dropped, flagged = Counter(), Counter(), [], 0
    gold_only = [(GoldTranscriptASR(), "gold")]
    work = [(s, sources) for s in chosen] + [(s, gold_only) for s in extras]
    asr_skipped = len(extras) * sum(1 for _, source in sources if source != "gold")
    with open(out, "w", encoding="utf-8") as f:
        for sample, sample_sources in work:
            for i, (asr, source) in enumerate(sample_sources):
                task = build_task(sample, asr, catalog, args.data_root, "train")  # CachedASR: KeyError on a missing clip
                if "env_id" in sample:  # the base sample must not come back as its own few-shot example
                    task = dataclasses.replace(task, id=env_id(sample))
                env = zaic_tools.ToolEnv("train", env_id(sample), root=args.data_root)  # fresh env per record
                lao = (args.leave_app_out is not None
                       and random.Random(f"{args.seed}:{sample['id']}:{i}:lao").random() < args.leave_app_out)
                rec = replay(sample, task, env, args.prompt, args.examples, bank, args.date_hints,
                             thoughts[sample["id"]]["steps"] if args.thoughts else None, leave_gold_apps_out=lao)
                if rec is None:
                    dropped.append(f"{sample['id']}:{source}")
                    continue
                rec["source"] = source
                if args.leave_app_out is not None:  # no key without the option, so default records are unchanged
                    rec["leave_app_out"] = lao
                    flagged += lao
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written[source] += 1
                if i == 0:
                    kinds[label_kind(sample["label"]["tool_calls"])] += 1
                    for k in range(args.mask):
                        masked = mask_record(rec, random.Random(f"{args.seed}:{sample['id']}:{k}"))
                        if args.perturb:
                            masked = perturb_record(masked, random.Random(f"{args.seed}:{sample['id']}:{k}:p"))
                        f.write(json.dumps(masked, ensure_ascii=False) + "\n")
                        written["masked"] += 1
                    if remix_table is not None and applies(rec, remix_table):  # no copy of a record the table leaves alone
                        remixed = remix_record(rec, remix_table, random.Random(f"{args.seed}:{sample['id']}:rx"))
                        f.write(json.dumps(remixed, ensure_ascii=False) + "\n")
                        written["remixed"] += 1

    print(f"-> {out}: {sum(written.values())} records {dict(written)}, first source by kind {dict(kinds)}, "
          f"{len(dropped)} replays dropped" + (f" {dropped[:20]}" if dropped else "")
          + (f", {len(extras)} extra samples ({asr_skipped} asr skipped)" if extras else "")
          + (f", {len(missing)} samples without sentences" + (f" {missing[:20]}" if missing else "")
             if args.thoughts else "")
          + (f", {flagged} records with the gold apps left out of the examples" if args.leave_app_out is not None else ""),
          flush=True)
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
