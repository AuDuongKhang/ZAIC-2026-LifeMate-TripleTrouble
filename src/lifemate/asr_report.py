"""Score ASR caches against train audio_text: one row per cache (clips, WER, CER, p50/p95 seconds per clip).

    python -m lifemate.asr_report --split train pho-medium-fp16 pho-large-fp16

Also writes asr_cache/<split>/report.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .asr.cached import cache_path, cache_root, read_cache
from .asr.wer import cer, wer
from .data import default_data_root, load_samples
from .transcribe import percentile

HEADER = "| cache | clips | WER | CER | p50 s | p95 s |\n|---|---|---|---|---|---|"


def references(samples: list[dict]) -> dict[str, str]:
    """{clip path as in the sample: gold transcript} for final requests and history user turns."""
    refs = {}
    for s in samples:
        refs[s["audio"]] = s["audio_text"]
        for t in s["history"]:
            if t["role"] == "user" and t.get("audio"):
                refs[t["audio"]] = t["audio_text"]
    return refs


def report_row(name: str, rows: dict[str, dict], refs: dict[str, str]) -> dict:
    scored = [r for clip, r in rows.items() if clip in refs]
    n = len(scored)
    secs = [r.get("seconds", 0.0) for r in scored]
    return {"cache": name, "clips": n,
            "wer": round(sum(wer(refs[r["audio"]], r["text"]) for r in scored) / n, 4) if n else None,
            "cer": round(sum(cer(refs[r["audio"]], r["text"]) for r in scored) / n, 4) if n else None,
            "p50": round(percentile(secs, 0.5), 3), "p95": round(percentile(secs, 0.95), 3)}


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+", help="cache names under asr_cache/<split>/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)

    samples = load_samples(args.data_root, args.split)
    if not samples or "audio_text" not in samples[0]:
        raise SystemExit(f"split {args.split!r} has no transcripts; only train can be scored")
    refs = references(samples)
    rows = []
    for name in args.caches:
        path = cache_path(args.split, name)
        if not path.exists():
            raise SystemExit(f"{path} not found")
        rows.append(report_row(name, read_cache(path), refs))
    lines = [HEADER] + [f"| {r['cache']} | {r['clips']} | {r['wer']} | {r['cer']} | {r['p50']} | {r['p95']} |" for r in rows]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = cache_root() / args.split / "report.md"
    out.write_text(text, encoding="utf-8")
    print(f"-> {out}")
    return rows


if __name__ == "__main__":
    main()
