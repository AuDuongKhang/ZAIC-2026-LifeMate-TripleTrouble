"""Check a context-prompt ASR cache: WER on all clips and on final requests, and how often it copies the prompt.

    python -m lifemate.asr_prompt_check pho-large-ft-b-context --base pho-large-ft-b

A clip counts as an echo when its transcript has a run of --ngram words that is in the sample's context prompt but not
in the gold transcript. Run the base cache through the same count: context words can appear by chance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .asr.cached import cache_path, read_cache, sample_clips
from .asr.context import context_prompt
from .asr.wer import normalize, wer
from .asr_report import references
from .data import default_data_root, load_samples


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = normalize(text)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def echo(hypothesis: str, reference: str, prompt: str, n: int = 4) -> set[tuple[str, ...]]:
    """Runs of n words the transcript shares with the prompt that the speaker did not say."""
    return (ngrams(hypothesis, n) & ngrams(prompt, n)) - ngrams(reference, n)


def check(rows: dict[str, dict], samples: list[dict], n: int = 4) -> dict:
    refs, wers, finals, echoes = references(samples), [], [], []
    for sample in samples:
        prompt = context_prompt(sample)
        for clip in sample_clips(sample):
            if clip not in rows:
                continue
            text = rows[clip]["text"]
            score = wer(refs[clip], text)
            wers.append(score)
            if clip == sample["audio"]:
                finals.append(score)
            if echo(text, refs[clip], prompt, n):
                echoes.append({"audio": clip, "reference": refs[clip], "text": text})
    return {"clips": len(wers), "wer": round(sum(wers) / max(len(wers), 1), 4),
            "final_wer": round(sum(finals) / max(len(finals), 1), 4), "echo_clips": len(echoes), "echoes": echoes}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("--base", default=None, help="the same model without the prompt")
    ap.add_argument("--ngram", type=int, default=4)
    ap.add_argument("--show", type=int, default=15, help="echo examples to print")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)

    samples = load_samples(args.data_root, "train")
    results = {}
    for name in [args.cache] + ([args.base] if args.base else []):
        path = cache_path("train", name)
        if not path.exists():
            raise SystemExit(f"{path} not found")
        results[name] = check(read_cache(path), samples, args.ngram)
        r = results[name]
        print(f"{name}: clips {r['clips']}, WER {r['wer']}, final requests WER {r['final_wer']}, "
              f"echo clips {r['echo_clips']}")
    for row in results[args.cache]["echoes"][: args.show]:
        print(f"{row['audio']}\n  gold: {row['reference']}\n  asr : {row['text']}")
    return results


if __name__ == "__main__":
    main()
