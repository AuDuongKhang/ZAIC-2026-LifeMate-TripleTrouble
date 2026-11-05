"""Reasoning sentences for SFT: one short Vietnamese sentence per gold step, written by a local open model on molab
(see docs/superpowers/specs/2026-11-05-reasoning-lines-design.md). Development tooling only: nothing in the inference
path (agents/, infer.py, docker_entry.py) may import this module.

The writer sees, for each gold step, the context, the history with gold transcripts, the request, the tools used so
far and the correct next action, but never that action's result or anything later. Every sentence is checked
(`check_sentence`); one that fails is retried once with the failure named, and a sample that still fails is left out.

    python -m lifemate.sft.thoughts --name pilot1 --subset pool --limit 100 --revision <teacher sha> --hf-repo <dataset>

Output: train_records/thoughts/<name>.jsonl, one {"id", "teacher", "revision", "steps": [{"call", "thought"}]} per
sample, plus <name>.failed.jsonl for the samples left out. Both files are append-only and flushed after every sample,
so a run that dies can be restarted with the same arguments: finished samples are skipped, a torn last line is cut,
and a file written by another teacher or revision is refused. With --hf-repo the files are pulled from the private HF
dataset when the session has none, and uploaded every --upload-every samples as thoughts/<name>[.failed].jsonl.
Only the train split is ever loaded.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from ..agents.ask_guard import fold
from ..agents.prompting import tool_schema
from ..agents.react import CLOSING
from ..agents.refs import scalar_strings
from ..data import REPO_ROOT, SYS_TOOL, default_data_root, import_zaic_tools, load_catalog, load_samples
from ..evaluate import label_kind
from ..splits import load_splits
from ..utils import seed_everything

THOUGHTS_DIR = REPO_ROOT / "train_records" / "thoughts"
TEACHER = "Qwen/Qwen3.5-27B"
SUBSETS = ("pool", "dev", "holdout")  # never "all": holdout is written only after sft-r1 passes its gate
TOTAL_STEPS = 3652  # pool 2271 + dev 725 + holdout 656
MAX_WORDS = 30
MIN_NAME = 3  # tool, app and parameter names shorter than this are not banned (too many false hits)

WRITER_RULES = """You write the reasoning line of a voice assistant that acts through tools. You are shown one step of a conversation and the correct next action. Write the single sentence the assistant thinks right before that action.
Rules:
1. Write one sentence in Vietnamese, at most 30 words, on a single line.
2. Say what the request needs, what the earlier results show, or what is missing. Reason before the action: do not describe it as already done.
3. Use plain words. Never write tool names, app names or parameter names. Ids and values taken from the request or from results are fine.
4. Use only what the prompt shows. Never rely on the result of the next action or on anything later.
5. If no tool is called because the request is out of scope or the user cancelled it, say why nothing is called.
6. Reply with the sentence only: no quotes, no markup, no explanation."""
RETRY = "That sentence was rejected: {reason}. Write it again as one sentence that follows every rule. Reply with the sentence only."


def load_thoughts(path: Path) -> dict[str, dict]:
    """Rows of a thoughts file by sample id. A torn last line (a session that died mid-write) is ignored; a broken
    line anywhere else raises."""
    rows: dict[str, dict] = {}
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
        rows[row["id"]] = row
    return rows


def step_calls(sample: dict) -> list[dict | None]:
    """The gold calls, or [None] (one step that calls nothing) for an out-of-scope sample."""
    return list(sample["label"]["tool_calls"]) or [None]


def _tool_names(calls: list[dict]) -> list[str]:
    """Tools the calls involve, in order: each called tool, and the tool an ask is about."""
    names: list[str] = []
    for call in calls:
        names.append(call["name"])
        if call["name"] == SYS_TOOL and call["arguments"].get("tool_name"):
            names.append(call["arguments"]["tool_name"])
    return names


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE) is not None


def banned_words(sample: dict, catalog: dict, results: list) -> set[str]:
    """Names a sentence must not contain: for each tool of the sample (and the tool an ask is about) its full name,
    app prefix, suffix and top-level parameter names; for an out-of-scope sample also the installed app ids. Names
    under MIN_NAME letters are dropped, and so are names the user said (request, history) or that appear as a value
    in `results`: those words are the user's or the data's, not tool vocabulary. A key that only appears in a
    result's JSON structure stays banned."""
    names: set[str] = set()
    for tool in _tool_names(sample["label"]["tool_calls"]):
        names.add(tool)
        names.update(tool.split(".", 1))
        names.update(((catalog.get(tool) or {}).get("parameters") or {}).get("properties", {}))
    if not sample["label"]["tool_calls"]:
        names.update(sample["context"]["installed_apps"])
    user_turns = [t.get("audio_text") or t.get("text") or "" for t in sample["history"] if t["role"] == "user"]
    said = "\n".join([sample["audio_text"], *user_turns, *sorted(scalar_strings(results, set()))])
    return {n.lower() for n in names if len(n) >= MIN_NAME and not _has_word(said, n)}


def check_sentence(text: str, banned: set[str]) -> str | None:
    """Why `text` is not an acceptable reasoning sentence, or None."""
    text = text.strip()
    if not text:
        return "empty"
    if "<tool_call" in text or "</tool_call" in text:
        return "markup"
    if tuple(fold(text)) in CLOSING:
        return "closing text"
    if "\n" in text or len(re.findall(r"[.!?](?=\s|$)", text)) > 1:
        return "several sentences"
    if len(text.split()) > MAX_WORDS:
        return "too long"
    for word in sorted(banned):
        if _has_word(text, word):
            return f"names {word}"
    return None


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _history_text(history: list[dict]) -> str:
    lines = []
    for turn in history:
        if turn["role"] == "user":
            lines.append(f"User: {turn.get('audio_text') or turn.get('text') or ''}")
            continue
        lines.append(f"Assistant: {turn.get('text') or ''}".rstrip())
        for call, result in zip(turn.get("tool_calls") or [], turn.get("tool_results") or []):
            lines.append(f"  called {call['name']} {_dump(call['arguments'])} -> {_dump(result)}")
    return "\n".join(lines) or "(none)"


def step_prompt(sample: dict, catalog: dict, step: int, results: list) -> list[dict]:
    """Messages for one step: the writer's rules, then everything the assistant knows at that point and the correct
    next action. `results` holds the results of the calls before `step`. Tools of later steps never appear."""
    calls = step_calls(sample)
    context = sample["context"]
    shown: list[str] = []
    for tool in _tool_names([c for c in calls[: step + 1] if c]):
        if tool not in shown and tool in catalog:
            shown.append(tool)
    parts = [
        f"Current time: {context['current_time']}\nTimezone: {context['timezone']}\nLocation: {_dump(context.get('location'))}"
        f"\nUser profile: {_dump(context.get('user_profile'))}",
        f"Earlier conversation (already done):\n{_history_text(sample['history'])}",
        f"User's request: {sample['audio_text']}",
    ]
    if shown:
        parts.append("Tools involved:\n" + "\n".join(_dump(tool_schema(catalog[t])) for t in shown))
    done = [f"{i + 1}. {c['name']} {_dump(c['arguments'])} -> {_dump(r)}" for i, (c, r) in enumerate(zip(calls[:step], results))]
    parts.append("Actions already taken for this request, with their results:\n" + ("\n".join(done) or "(none)"))
    if calls[step] is None:
        parts.append("Installed apps: " + ", ".join(context["installed_apps"]) + "\nNext action: none. No tool is called "
                     "for this request: it is out of scope, or the user cancelled it. The assistant says so.")
    else:
        parts.append(f"Next action: {calls[step]['name']} {_dump(calls[step]['arguments'])}")
    parts.append("Write the one-sentence reasoning line that comes before the next action.")
    return [{"role": "system", "content": WRITER_RULES}, {"role": "user", "content": "\n\n".join(parts)}]


def write_sample(sample: dict, catalog: dict, env, llm, teacher: str, revision: str,
                 stats: Counter) -> tuple[dict | None, dict | None]:
    """(row, failure), exactly one None. Writes the sentence of each gold step in order, then runs that gold call in
    `env` to get the result the next step may rely on. `llm.generate(messages) -> str`. A sentence that fails its
    checks is retried once with the failure named; a second failure ends the sample (its later steps are not
    written) and is counted under stats["failed"] and stats["reason: <first word>"]."""
    steps: list[dict] = []
    results: list = []
    for i, call in enumerate(step_calls(sample)):
        messages = step_prompt(sample, catalog, i, results)
        banned = banned_words(sample, catalog, results)
        text = llm.generate(messages).strip()
        reason = check_sentence(text, banned)
        if reason is None:
            stats["first_try"] += 1
        else:
            retry = messages + [{"role": "assistant", "content": text},
                                {"role": "user", "content": RETRY.format(reason=reason)}]
            text = llm.generate(retry).strip()
            reason = check_sentence(text, banned)
            if reason is not None:
                stats["failed"] += 1
                stats[f"reason: {reason.split()[0]}"] += 1
                return None, {"id": sample["id"], "step": i, "reason": reason, "text": text}
            stats["retried"] += 1
        steps.append({"call": call["name"] if call else None, "thought": text})
        if call:
            results.append(env.call(call["name"], call["arguments"]))
    return {"id": sample["id"], "teacher": teacher, "revision": revision, "steps": steps}, None


def clean_completion(text: str, stop_strings: set[str]) -> str:
    """The teacher's reply without end-of-turn markers and without any <think> block."""
    for s in stop_strings:
        text = text.replace(s, "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.split("</think>")[-1].strip()


def teacher_loader(architectures: list[str] | None) -> str:
    """The transformers auto class that loads a checkpoint with its own parameter names. A multimodal checkpoint
    (`...ForConditionalGeneration`, like Qwen3.5) keeps its text weights under `model.language_model.*`; the causal-LM
    auto class would build a text-only model with other names rather than raise, so use the class it was saved from."""
    if any(a.endswith("ForConditionalGeneration") for a in architectures or []):
        return "AutoModelForImageTextToText"
    return "AutoModelForCausalLM"


class Teacher:
    """The local teacher: an HF chat model with thinking off, greedy decoding. Heavy imports stay inside __init__ so
    the module (and its tests) load without torch."""

    def __init__(self, model: str, revision: str, max_new_tokens: int = 128, dtype: str = "bfloat16"):
        import torch
        import transformers

        self.torch, self.max_new_tokens = torch, max_new_tokens
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, revision=revision)
        kwargs = {"dtype": getattr(torch, dtype), "revision": revision}
        config = transformers.AutoConfig.from_pretrained(model, revision=revision)
        loader = teacher_loader(config.architectures)
        try:
            self.model = getattr(transformers, loader).from_pretrained(model, **kwargs)
        except ValueError:  # the auto class does not know this checkpoint: try the other one
            other = "AutoModelForCausalLM" if loader == "AutoModelForImageTextToText" else "AutoModelForImageTextToText"
            self.model = getattr(transformers, other).from_pretrained(model, **kwargs)
        self.model = self.model.to("cuda").eval()
        self.stop_strings = {t for t in (self.tokenizer.eos_token, self.tokenizer.pad_token, "<|im_end|>", "<|endoftext|>") if t}
        seed_everything(0)

    def generate(self, messages: list[dict]) -> str:
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                                    return_tensors="pt", return_dict=True).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                                      temperature=None, top_p=None, top_k=None)
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        return clean_completion(text, self.stop_strings)


def _trim_torn(path: Path) -> None:
    """Cut a last line that a dead session left half written (or end a complete one with a newline)."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, _, tail = data.rpartition(b"\n")
    try:
        json.loads(tail)
    except ValueError:
        path.write_bytes(head + b"\n" if head else b"")
    else:
        path.write_bytes(data + b"\n")


def _pull_hf_copy(hf_repo: str, paths: list[Path]) -> None:
    """Fetch thoughts/<file> from the private HF dataset into a fresh session. A file HF does not have is fine (a new
    run); any other error stops the run, so a fresh local file can never overwrite the HF copy on the next upload.
    Being offline (LocalEntryNotFoundError) is such an error, not an absent file."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    with tempfile.TemporaryDirectory() as tmp:
        for path in paths:
            try:
                got = hf_hub_download(repo_id=hf_repo, filename=f"thoughts/{path.name}", repo_type="dataset", local_dir=tmp)
            except LocalEntryNotFoundError:
                raise
            except EntryNotFoundError:
                continue
            shutil.copyfile(got, path)
            print(f"pulled {hf_repo}/thoughts/{path.name}", flush=True)


def _upload(hf_repo: str, paths: list[Path]) -> None:
    """Upload the non-empty files; a failure is printed and the run goes on (the next upload retries)."""
    from huggingface_hub import HfApi

    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"thoughts/{path.name}", repo_id=hf_repo,
                                repo_type="dataset")
        except Exception as e:  # noqa: BLE001 - a failed upload must not kill a long run
            print(f"upload of {path.name} to {hf_repo} failed: {type(e).__name__}: {e}", flush=True)


def main(argv: list[str] | None = None, llm=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", choices=SUBSETS, required=True)
    ap.add_argument("--revision", required=True, help="teacher revision (commit sha), pinned in every row")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--limit", type=int, default=None, help="K samples drawn at random (seed 0) from the subset")
    ap.add_argument("--hf-repo", default=None, help="private HF dataset holding thoughts/<name>.jsonl")
    ap.add_argument("--upload-every", type=int, default=200, help="samples between uploads")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--out-dir", type=Path, default=THOUGHTS_DIR)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}.jsonl"
    failed_path = args.out_dir / f"{args.name}.failed.jsonl"
    if args.hf_repo and not path.exists():
        _pull_hf_copy(args.hf_repo, [path, failed_path])
    _trim_torn(path)
    _trim_torn(failed_path)
    written = load_thoughts(path) if path.exists() else {}
    for row in written.values():
        if row.get("teacher") != args.teacher:
            raise SystemExit(f"{path} was written by teacher {row.get('teacher')}, not {args.teacher}: "
                             "one file never mixes teachers, use another --name")
        if row.get("revision") != args.revision:
            raise SystemExit(f"{path} was written at revision {row.get('revision')}, not {args.revision}: "
                             "one file never mixes revisions, use another --name")
    failed = set(load_thoughts(failed_path)) if failed_path.exists() else set()
    if llm is None:  # only now: a refused or broken resume must not cost a model load
        llm = Teacher(args.teacher, args.revision)

    zaic_tools = import_zaic_tools(args.data_root, "train")
    samples = load_samples(args.data_root, "train")
    catalog = load_catalog(args.data_root, "train")
    wanted = set(load_splits()[args.subset])
    chosen = [s for s in samples if s["id"] in wanted]
    if args.limit:
        chosen = random.Random(0).sample(chosen, min(args.limit, len(chosen)))
    todo = [s for s in chosen if s["id"] not in written and s["id"] not in failed]
    print(f"{args.subset}: {len(chosen)} samples, {len(chosen) - len(todo)} already done, {len(todo)} to write", flush=True)

    stats, kinds = Counter(), Counter()
    n_written = n_failed = since_upload = 0
    start = time.monotonic()
    with open(path, "a", encoding="utf-8") as out, open(failed_path, "a", encoding="utf-8") as bad:
        for sample in todo:
            env = zaic_tools.ToolEnv("train", sample["id"], root=args.data_root)  # one fresh env per sample
            row, failure = write_sample(sample, catalog, env, llm, args.teacher, args.revision, stats)
            if row is not None:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                kinds[label_kind(sample["label"]["tool_calls"])] += 1
                n_written += 1
            else:
                bad.write(json.dumps(failure, ensure_ascii=False) + "\n")
                bad.flush()
                n_failed += 1
            since_upload += 1
            if args.hf_repo and since_upload >= args.upload_every:
                _upload(args.hf_repo, [path, failed_path])
                since_upload = 0
    if args.hf_repo and since_upload:
        _upload(args.hf_repo, [path, failed_path])

    seconds = time.monotonic() - start
    attempted = stats["first_try"] + stats["retried"] + stats["failed"]
    pass_rate = f"{(stats['first_try'] + stats['retried']) / attempted:.1%}" if attempted else "n/a"
    per_step = seconds / attempted if attempted else None
    reasons = {k[len("reason: "):]: v for k, v in stats.items() if k.startswith("reason: ")}
    print(f"-> {path}: wrote {n_written}, failed {n_failed}; first_try {stats['first_try']}, retried {stats['retried']}, "
          f"failed {stats['failed']}, pass rate {pass_rate}, reasons {reasons}, kinds {dict(kinds)}; "
          + (f"{per_step:.2f} s per step, {per_step * TOTAL_STEPS / 3600:.1f} hours for {TOTAL_STEPS} steps"
             if per_step is not None else f"no steps run, hours for {TOTAL_STEPS} steps unknown"), flush=True)
    return path


if __name__ == "__main__":
    main()
