"""Let the LLM repair misheard words in the final request of an ASR cache, and write the result as a new cache.

    python -m lifemate.asr_fix --source pho-large-ft-b-beam5 --name beam5-fix --model <owner>/lifemate-sft-v2 --subset eval
    python -m lifemate.infer --split train --agent-preset sft-v2-refs --asr cached --asr-kw name=beam5-fix --subset holdout

Each prompt holds what the agent may see at inference: installed app names, the user's profile, earlier turns (their ASR
text, never gold) and the transcript. History clips are copied unchanged. An answer is dropped, keeping the transcript,
when it is empty, holds markup or several lines, or differs from the transcript by more than --max-change (word edits per
word). The LLM must be the one the system ships, since a second model would break the 5B limit. On train it also writes
<name>.fix.md: final-request WER before and after by subset, with the clips that improved and got worse. Tune on dev only.
Finished clips are skipped, so an interrupted run resumes. Greedy decoding, so a rerun gives the same text.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .asr.cached import cache_path, read_cache, sample_clips
from .asr.wer import wer
from .data import default_data_root, load_samples
from .splits import SUBSETS, load_splits
from .transcribe import percentile
from .utils import seed_everything

SYSTEM = """You correct Vietnamese speech-recognition transcripts for a voice assistant that controls phone apps.
The transcript may contain misheard words: real words that sound like the right one but make no sense here.
Use the installed apps, the profile and the earlier turns to decide what the user meant.

Rules:
- Change only words that are clearly misheard. If you are not sure, keep the word.
- Keep names, numbers, word order, fillers and regional words exactly as they are unless they are clearly misheard.
- Do not answer the request, add words or explain.
- Output only the corrected transcript, on one line. If nothing needs correcting, output it unchanged.

Examples:
Transcript: Nhắc tui mua hộp sửa tươi chiều nay nha
Output: Nhắc tui mua hộp sữa tươi chiều nay nha
Transcript: Huỷ cái lịch hộp với chị Lan sáng mai giùm em
Output: Huỷ cái lịch họp với chị Lan sáng mai giùm em
Transcript: Mở bài của Hoài Lâm lên nghe đi em
Output: Mở bài của Hoài Lâm lên nghe đi em"""


def build_messages(sample: dict, transcript: str, rows: dict[str, dict], app_names: dict[str, str]) -> list[dict]:
    """Chat messages for one final request; earlier user turns come from the ASR cache, so no gold text gets in."""
    turns = []
    for turn in sample["history"][-6:]:
        if turn["role"] == "user":
            row = rows.get(turn.get("audio"))
            if row:
                turns.append("User: " + row["text"])
        elif turn.get("text"):
            turns.append("Assistant: " + turn["text"])
    apps = [app_names.get(a, a) for a in sample["context"]["installed_apps"]]
    parts = ["Installed apps: " + ", ".join(apps),
             "Profile: " + json.dumps(sample["context"].get("user_profile") or {}, ensure_ascii=False)]
    if turns:
        parts.append("Earlier turns:\n" + "\n".join(turns))
    parts.append("Transcript: " + transcript)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def clean_answer(transcript: str, answer: str, max_change: float) -> tuple[str, str]:
    """(text to keep, outcome), where outcome is same, changed, or the reason the answer was dropped."""
    text = answer.strip()
    if text.lower().startswith("output:"):
        text = text[len("output:"):].strip()
    if not text:
        return transcript, "dropped: empty"
    if "<" in text or "\n" in text:
        return transcript, "dropped: markup or several lines"
    if wer(transcript, text) > max_change:
        return transcript, "dropped: too different"
    return (transcript, "same") if text == transcript else (text, "changed")


def fix_samples(samples: list[dict], rows: dict[str, dict], done: dict[str, dict], llm, app_names: dict[str, str],
                max_change: float, out_file) -> dict[str, dict]:
    """Write one line per clip not in `done` and return every written or finished line."""
    result = dict(done)

    def write(row: dict) -> None:
        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_file.flush()
        result[row["audio"]] = row

    for i, sample in enumerate(samples, 1):
        for clip in sample_clips(sample):
            if clip in result:
                continue
            if clip not in rows:
                raise KeyError(f"{sample['id']}: no cached transcript for {clip}")
            source = rows[clip]
            if clip != sample["audio"]:
                write({**source, "original": source["text"], "outcome": "history"})
                continue
            t0 = time.perf_counter()
            answer = llm.generate(build_messages(sample, source["text"], rows, app_names), [])
            text, outcome = clean_answer(source["text"], answer, max_change)
            write({"audio": clip, "text": text, "original": source["text"], "outcome": outcome,
                   "seconds": round(time.perf_counter() - t0, 3)})
        if i % 50 == 0 or i == len(samples):
            print(f"[{i}/{len(samples)}] samples", flush=True)
    return result


def report(name: str, samples: list[dict], rows: dict[str, dict], groups: dict[str, set[str]], top: int = 15) -> str:
    """Final-request WER before and after, by group of sample ids; needs audio_text (train)."""
    lines = [f"# LLM transcript fix: {name}", "", "| subset | clips | WER before | WER after | better | worse | same text |",
             "|---|---|---|---|---|---|---|"]
    moved = []
    for group, ids in groups.items():
        before, after, better, worse, same = [], [], 0, 0, 0
        for s in samples:
            row = rows.get(s["audio"])
            if s["id"] not in ids or row is None:
                continue
            b, a = wer(s["audio_text"], row["original"]), wer(s["audio_text"], row["text"])
            before.append(b)
            after.append(a)
            better += a < b
            worse += a > b
            same += row["text"] == row["original"]
            if a != b:
                moved.append((a - b, s["id"], s["audio_text"], row["original"], row["text"]))
        if before:
            lines.append(f"| {group} | {len(before)} | {sum(before) / len(before):.4f} | {sum(after) / len(after):.4f} | "
                         f"{better} | {worse} | {same} |")
    finals = [rows[s["audio"]] for s in samples if s["audio"] in rows]
    outcomes: dict[str, int] = {}
    for row in finals:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    secs = [r.get("seconds", 0.0) for r in finals]
    lines += ["", f"Outcomes: {outcomes}", f"Seconds per final request: p50 {percentile(secs, 0.5):.2f}, "
              f"p95 {percentile(secs, 0.95):.2f}, max {max(secs, default=0.0):.2f}"]
    for title, picked in (("Got worse", sorted(moved, reverse=True)[:top]), ("Improved", sorted(moved)[:top])):
        lines += ["", f"## {title}", ""]
        for diff, sid, gold, original, text in picked:
            if (diff > 0) == (title == "Got worse"):
                lines += [f"- `{sid}` WER {diff:+.3f}", f"  - gold: {gold}", f"  - asr:  {original}", f"  - fix:  {text}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="ASR cache to read")
    ap.add_argument("--name", required=True, help="new cache to write")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", choices=(*SUBSETS, "eval"), default=None, help="train only; eval = dev + holdout")
    ap.add_argument("--limit", type=int, default=None, help="first N samples only")
    ap.add_argument("--max-change", type=float, default=0.3, help="drop answers with more word edits per word than this")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    args = ap.parse_args(argv)

    rows = read_cache(cache_path(args.split, args.source))
    if not rows:
        raise SystemExit(f"{cache_path(args.split, args.source)} not found or empty")
    samples = load_samples(args.data_root, args.split)
    splits = load_splits() if args.split == "train" else {}
    if args.subset:
        if args.split != "train":
            ap.error("--subset needs --split train")
        wanted = set(splits["dev"]) | set(splits["holdout"]) if args.subset == "eval" else set(splits[args.subset])
        samples = [s for s in samples if s["id"] in wanted]
    samples = samples[: args.limit]
    with open(args.data_root / args.split / f"{args.split}_tools_catalog.json", encoding="utf-8") as f:
        app_names = {a["id"]: a["name_vi"] for a in json.load(f)["apps"]}

    out = cache_path(args.split, args.name)
    done = read_cache(out)
    todo = [s for s in samples if any(c not in done for c in sample_clips(s))]
    print(f"{args.name}: {len(todo)} samples to fix ({len(samples) - len(todo)} already done)", flush=True)
    result = done
    if todo:
        from .agents.llm import HFChatLLM

        llm = HFChatLLM(args.model, max_new_tokens=args.max_new_tokens)
        seed_everything(0)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            result = fix_samples(todo, rows, done, llm, app_names, args.max_change, f)
    if args.split == "train":
        groups = {g: set(splits[g]) for g in ("dev", "holdout", "pool")}
        text = report(args.name, samples, result, groups)
        out.with_suffix(".fix.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
    return out


if __name__ == "__main__":
    main()
