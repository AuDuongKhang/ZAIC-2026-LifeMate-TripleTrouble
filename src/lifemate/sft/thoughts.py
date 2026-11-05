"""Reasoning sentences for SFT: one short Vietnamese sentence per gold step, written by a local open model on molab
(see docs/superpowers/specs/2026-11-05-reasoning-lines-design.md). Development tooling only: nothing in the inference
path (agents/, infer.py, docker_entry.py) may import this module.

The writer sees, for each gold step, the context, the history with gold transcripts, the request, the tools used so
far and the correct next action, but never that action's result or anything later. Every sentence is checked
(`check_sentence`); one that fails is retried once with the failure named, and a sample that still fails is left out.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from ..agents.ask_guard import fold
from ..agents.prompting import tool_schema
from ..agents.react import CLOSING
from ..agents.refs import scalar_strings
from ..data import SYS_TOOL

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
