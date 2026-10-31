"""Transcribe every clip (final request + history user turns) of a split once into asr_cache/<split>/<name>.jsonl.

    python -m lifemate.transcribe --split train --asr whisper --asr-kw model=vinai/PhoWhisper-large --name pho-large-fp16 --subset eval
    python -m lifemate.infer --split train --agent react --asr cached --asr-kw name=pho-large-fp16 --subset dev

Clips already in the cache are skipped, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .asr import BACKENDS, build_asr
from .asr.cached import cache_path, read_cache, sample_clips
from .data import audio_path, default_data_root, load_samples
from .splits import SUBSETS, load_splits
from .utils import parse_kw, seed_everything


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--asr", required=True, choices=sorted(set(BACKENDS) - {"cached"}))
    ap.add_argument("--asr-kw", action="append", metavar="KEY=VALUE", help="ASR constructor option (repeatable)")
    ap.add_argument("--name", required=True, help="cache name, e.g. pho-large-fp16")
    ap.add_argument("--subset", choices=(*SUBSETS, "eval"), default=None, help="train only; eval = dev + holdout")
    ap.add_argument("--limit", type=int, default=None, help="first N samples only")
    args = ap.parse_args(argv)

    samples = load_samples(args.data_root, args.split)
    if args.subset:
        if args.split != "train":
            ap.error("--subset needs --split train")
        splits = load_splits()
        wanted = set(splits["dev"]) | set(splits["holdout"]) if args.subset == "eval" else set(splits[args.subset])
        samples = [s for s in samples if s["id"] in wanted]
    if args.limit:
        samples = samples[: args.limit]

    out = cache_path(args.split, args.name)
    done = read_cache(out)
    todo = [(s, c) for s in samples for c in sample_clips(s) if c not in done]
    print(f"{args.name}: {len(todo)} clips to transcribe ({len(done)} already cached)", flush=True)
    if not todo:
        return out
    asr = build_asr(args.asr, **parse_kw(args.asr_kw))
    seed_everything(0)
    out.parent.mkdir(parents=True, exist_ok=True)
    times = []
    with open(out, "a", encoding="utf-8") as f:
        for i, (sample, clip) in enumerate(todo, 1):
            t0 = time.perf_counter()
            text = asr.transcribe(audio_path(args.data_root, args.split, clip), sample)
            times.append(time.perf_counter() - t0)
            f.write(json.dumps({"audio": clip, "text": text, "seconds": round(times[-1], 3)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] mean {sum(times) / len(times):.3f}s/clip", flush=True)
    print(f"-> {out}  mean {sum(times) / len(times):.3f}s  p95 {percentile(times, 0.95):.3f}s per clip")
    return out


if __name__ == "__main__":
    main()
