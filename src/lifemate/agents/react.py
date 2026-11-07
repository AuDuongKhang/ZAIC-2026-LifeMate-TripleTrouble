"""Training-free tool-use agent: the LLM proposes calls in its native format, env.validate gates every call
before env.call (rejected calls are fed back, never sent), results go back into the conversation."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from ..data import SYS_TOOL, default_data_root
from ..task import Task
from .ask_guard import fold
from .base import Agent
from .date_fix import fix_calls
from .fewshot import ExampleBank, gold_apps, render_examples
from .llm import ChatLLM, HFChatLLM
from .parsing import BLOCK, parse_tool_calls
from .prompting import build_messages, system_message, tool_schema
from .prompts import DEFAULT_PROMPT, FINAL_TEXT, REFUSAL_TEXT
from .refs import scalar_strings, unseen_refs
from .refusal import matching_tools

EMPTY_HINT = ("\nNo results. Try once more with other values: another date (up to 7 days away), fewer optional filters "
              "or a shorter query. If that also finds nothing, reply in one short sentence without calling any tool.")
EMPTY_REPEAT = "Already called and it returned nothing. Change the arguments as suggested, or stop."
REFUSAL_CHECK = ("Check again: the installed tool(s) {names} may handle this request. If one does, call it. If the "
                 "request is out of scope or the user cancelled it, reply in one short sentence without calling any tool.")
CLOSING = {tuple(fold(FINAL_TEXT)), tuple(fold(REFUSAL_TEXT))}  # closing replies, folded
INVALID = ("invalid",)  # vote key of a proposal that does not parse


def leaked_param(spec: dict | None, arguments: dict) -> str | None:
    """The first fuzzy string parameter whose value is a closing reply (the model wrote it as message content)."""
    props = ((spec or {}).get("parameters") or {}).get("properties", {})
    for name, value in arguments.items():
        if isinstance(value, str) and props.get(name, {}).get("x-match") == "fuzzy" and tuple(fold(value)) in CLOSING:
            return name
    return None


def vote_key(call: dict, spec: dict | None) -> tuple:
    """What a proposal must agree on to count as the same vote: the tool and its arguments, without the free-text
    (x-match: fuzzy) ones, which are scored by token F1; for an ask, the tool asked about and the missing params."""
    def dump(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    args = call["arguments"]
    if call["name"] == SYS_TOOL:
        params = args.get("params")
        return SYS_TOOL, dump(args.get("tool_name")), dump(sorted(params, key=str) if isinstance(params, list) else params)
    props = ((spec or {}).get("parameters") or {}).get("properties", {})
    return call["name"], tuple(sorted((k, dump(v)) for k, v in args.items()
                                      if (props.get(k) or {}).get("x-match") != "fuzzy"))


def decides(calls: list[dict], specs: dict[str, dict]) -> bool:
    """The calls include a write or an ask: the step where the prompt vote happens, and the only kind of proposal
    that can win it."""
    return any(c["name"] == SYS_TOOL or (specs.get(c["name"]) or {}).get("kind") == "write" for c in calls)


def output_key(text: str | None, specs: dict[str, dict]) -> tuple[tuple, bool]:
    """(key, eligible) of one proposal. The key is its calls' vote keys in order, INVALID when a block does not parse
    (or there is no text). Only a valid proposal with a write or an ask is eligible to win; plain answers, read-only
    and invalid proposals are recorded but never counted, so the vote cannot turn an action into an answer or a read."""
    if text is None:
        return INVALID, False
    calls, errors = parse_tool_calls(text)
    if errors:
        return INVALID, False
    return tuple(vote_key(c, specs.get(c["name"])) for c in calls), decides(calls, specs)


def is_empty_result(result) -> bool:
    """A search that found nothing: `[]`, or a dict with at least one list value where every list value is empty."""
    if isinstance(result, list):
        return not result
    lists = [v for v in result.values() if isinstance(v, list)] if isinstance(result, dict) else []
    return bool(lists) and not any(lists)


class ReActAgent(Agent):
    name = "react"

    def __init__(self, llm: ChatLLM | None = None, model: str = "Qwen/Qwen3-4B-Instruct-2507", max_steps: int = 8,
                 max_invalid: int = 2, time_budget_s: float = 20.0, min_step_s: float = 3.0,
                 prompt_version: str = DEFAULT_PROMPT, check_refs: bool = False,
                 examples: int = 0, split: str = "train", data_root: Path | None = None, bank: ExampleBank | None = None,
                 clock: Callable[[], float] = time.perf_counter, revision: str | None = None,
                 date_hints: bool = False, empty_hint: bool = False,
                 refusal_check: bool = False, refusal_min_hits: int = 1,
                 leak_guard: bool = False, vote_prompts: int = 0, vote_apply: bool = True,
                 date_fix: bool = False, analog_weight: float = 0.0, max_new_tokens: int = 384, num_beams: int = 1,
                 leave_gold_apps_out: bool = False):
        """time_budget_s counts from task.started_at (sample start, ASR included) when the runner sets it.
        A step only starts with at least min_step_s left, and its generation is capped at the time remaining.
        check_refs holds back calls whose x-ref ids were never seen in a tool result (off by default: with it the
        model dodged the check, e.g. calling by phone_number instead of asking which contact).
        examples=k puts the k most similar solved training requests into the system prompt. split is the split being
        run (infer sets it): on train the bank is the experiment pool, on any other split every train sample.
        revision pins the HF model repo commit (shipped presets set it, so a later push cannot change the system).
        date_hints appends a weekday table (lifemate.agents.dates) to the system prompt.
        empty_hint adds a retry hint to the tool message after an empty read; only the message changes, never the calls.
        refusal_check: when the model's first output is a plain answer and an installed tool shares at least
        refusal_min_hits word bigrams with the request (agents/refusal.py), ask it once more before accepting the
        refusal. Never used when SFT records are rendered (ReplayLLM has no second answer, so the default is off there).
        leak_guard holds back a call whose fuzzy string parameter is the closing text ("Đã xong." or the refusal), which
        the model sometimes writes as the message content; the call is rejected like an invalid one, never sent.
        vote_prompts=K (0 = off) votes once per sample, at the first step whose output has a write call or an ask:
        K-1 more proposals for that step come from the same conversation with the system message showing the next
        examples (variant i: ranks i*k+1 .. i*k+k), and the majority proposal (at least 2 votes, ties to the original)
        is used instead (vote_apply=False only records the vote). Only proposals with a write or an ask can win;
        plain answers, read-only and invalid proposals are recorded but not counted, so the original is the fallback. Greedy decoding stays; only the used proposal is
        sent. The vote is skipped when less than K * min_step_s is left after the original proposal.
        date_fix replaces the date of a call's single date argument with the one the request's only date phrase
        resolves to (agents/date_fix.py), before the call is validated; no extra call is ever sent.
        analog_weight > 0 makes the example selection also reward examples whose tools have the shape of an installed
        tool of an app the bank has no example of (ExampleBank.select); 0 is plain BM25.
        max_new_tokens and num_beams go to the LLM the agent builds (an `llm` passed in keeps its own); beam search
        stays deterministic but costs time.
        leave_gold_apps_out (SFT rendering and the train-only diagnostic, never a real run) leaves the apps of the
        task's gold calls out of the examples; it reads Task.label and raises ValueError without one."""
        if vote_prompts and not examples:
            raise ValueError("vote_prompts needs examples (the variants differ only in their examples)")
        llm_kw = {"revision": revision, **({"max_new_tokens": max_new_tokens} if max_new_tokens != 384 else {}),
                  **({"num_beams": num_beams} if num_beams != 1 else {})}  # only what differs from the LLM's defaults
        self.llm = llm if llm is not None else HFChatLLM(model, **llm_kw)
        self.max_steps, self.max_invalid, self.clock = max_steps, max_invalid, clock
        self.time_budget_s, self.min_step_s = time_budget_s, min_step_s
        self.prompt_version, self.check_refs, self.examples = prompt_version, check_refs, examples
        self.bank, self.date_hints, self.empty_hint = bank, date_hints, empty_hint
        self.refusal_check, self.refusal_min_hits = refusal_check, refusal_min_hits
        self.leak_guard, self.date_fix = leak_guard, date_fix
        self.vote_prompts, self.vote_apply = vote_prompts, vote_apply
        self.analog_weight = analog_weight
        self.leave_gold_apps_out = leave_gold_apps_out
        if self.bank is None and examples:
            self.bank = ExampleBank.from_data(data_root or default_data_root(), split)
        # Recorded in meta.json, so every RESULTS.md row traces back to its prompt and settings.
        self.info = {"model": model if llm is None else type(llm).__name__, "revision": revision,
                     "prompt": prompt_version, "check_refs": check_refs, "max_steps": max_steps, "time_budget_s": time_budget_s,
                     "examples": examples, "bank_size": len(self.bank) if self.bank is not None else 0,
                     "date_hints": date_hints, "empty_hint": empty_hint, "refusal_check": refusal_check, "refusal_min_hits": refusal_min_hits,
                     "leak_guard": leak_guard, "vote_prompts": vote_prompts, "vote_apply": vote_apply, "date_fix": date_fix,
                     "analog_weight": analog_weight, "max_new_tokens": max_new_tokens, "num_beams": num_beams,
                     "leave_gold_apps_out": leave_gold_apps_out}

    def run(self, task: Task, env) -> None:
        trace = self.last_trace = {"steps": [], "stop": None}  # None: a step raised
        start = task.started_at if task.started_at is not None else self.clock()
        if self.leave_gold_apps_out and task.label is None:
            raise ValueError("leave_gold_apps_out reads the gold label, but this task has no label")
        left_out = {"exclude_apps": gold_apps(task.label)} if self.leave_gold_apps_out else {}
        tools = [tool_schema(spec) for spec in task.tools]
        specs = {spec["name"]: spec for spec in task.tools}
        seen = scalar_strings([turn.get("tool_results") for turn in task.history], set())
        budget = self.llm.context_limit - self.llm.max_new_tokens
        examples_text = ""
        if self.bank is not None:
            analog = {"tools": task.tools, "analog_weight": self.analog_weight} if self.analog_weight else {}
            picked = self.bank.select(task.request, set(task.context["installed_apps"]), self.examples, exclude_id=task.id,
                                      **analog, **left_out)
            examples_text = render_examples(picked)
        messages = build_messages(task, lambda m: self.llm.count_tokens(m, tools), budget, self.prompt_version,
                                  examples_text, self.date_hints)
        done: set[str] = set()  # canonical (name, arguments) of calls already sent
        empty: set[str] = set()  # the subset of done whose result was empty (only tracked with empty_hint)
        invalid = 0
        checked = False  # the refusal check runs at most once, and only after the first output
        voted = not self.vote_prompts  # the prompt vote runs at most once, at the first write or ask
        if self.vote_prompts:
            trace["vote"] = None  # stays None when no step was voted on (reads only, or too little time)
        for _ in range(self.max_steps):
            remaining = self.time_budget_s - (self.clock() - start)
            if remaining < self.min_step_s:
                trace["stop"] = "time"
                return
            if self.llm.count_tokens(messages, tools) > budget:
                trace["stop"] = "tokens"
                return
            text = self.llm.generate(messages, tools, max_time=remaining)
            calls, errors = parse_tool_calls(text)
            if not voted and decides(calls, specs):
                voted = True
                text = self._vote(task, messages, tools, specs, text, start, budget)
                calls, errors = parse_tool_calls(text)
            rejected: list[str] = []
            trace["steps"].append({"output": text[:2000], "rejected": rejected})
            if not calls and not errors and self.refusal_check and not checked and len(trace["steps"]) == 1:
                checked = True
                names = matching_tools(task.request, task.tools, self.refusal_min_hits)
                if names:
                    trace["refusal_check"] = names
                    messages.append({"role": "assistant", "content": BLOCK.sub("", text).strip(), "tool_calls": []})
                    messages.append({"role": "user", "content": REFUSAL_CHECK.format(names=", ".join(names))})
                    continue
            if not calls and not errors:
                trace["stop"] = "no_call"  # a plain answer: finished, or nothing to do (out of scope)
                return
            if self.date_fix:  # before the assistant turn is written, so the model reads back the call that was sent
                fixed = fix_calls(calls, specs, task.request, task.context["current_time"])
                if fixed is not calls:
                    trace.setdefault("date_fixes", []).extend({"from": a["arguments"], "to": b["arguments"]}
                                                              for a, b in zip(calls, fixed) if a is not b)
                calls = fixed
            messages.append({"role": "assistant", "content": BLOCK.sub("", text).strip(), "tool_calls": [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
            for call in calls:
                key = json.dumps([call["name"], call["arguments"]], sort_keys=True, ensure_ascii=False)
                if key in done:
                    invalid += 1
                    rejected.append(f"{call['name']}: already called")
                    messages.append({"role": "tool", "content": EMPTY_REPEAT if key in empty
                                     else "Already called; reuse the earlier result."})
                    continue
                check = env.validate(call["name"], call["arguments"])
                if not check.get("ok"):
                    invalid += 1
                    rejected.append(f"{call['name']}: {check.get('error')}")
                    messages.append({"role": "tool", "content": f"Rejected: {check.get('error')}. Fix the call."})
                    continue
                leaked = leaked_param(specs.get(call["name"]), call["arguments"]) if self.leak_guard else None
                if leaked:
                    invalid += 1
                    rejected.append(f"{call['name']}: {leaked} repeats your closing reply")
                    messages.append({"role": "tool", "content": f"Not sent: {leaked} repeats your closing reply. "
                                     "Write the content the user asked for."})
                    continue
                unseen = unseen_refs(specs.get(call["name"]), call["arguments"], seen) if self.check_refs else []
                if unseen:
                    invalid += 1
                    rejected.append(f"{call['name']}: unknown id {', '.join(unseen)}")
                    messages.append({"role": "tool", "content": f"Not sent: unknown id {', '.join(unseen)}. Ids must "
                                     "come from tool results; call a read or search tool to look it up first."})
                    continue
                result = env.call(call["name"], call["arguments"])
                scalar_strings(result, seen)
                done.add(key)
                if call["name"] == SYS_TOOL:
                    trace["stop"] = "asked"
                    return  # asking the user ends the turn; later calls in the batch are dropped
                content = json.dumps(result, ensure_ascii=False)
                if self.empty_hint and is_empty_result(result):
                    empty.add(key)
                    content += EMPTY_HINT
                messages.append({"role": "tool", "content": content})
            if errors:
                invalid += 1
                rejected.extend(f"invalid JSON: {e}" for e in errors)
                messages.append({"role": "user", "content": "Your tool call was not valid JSON: " + "; ".join(errors)})
            if invalid > self.max_invalid:
                trace["stop"] = "invalid"
                return
        trace["stop"] = "max_steps"

    def _vote(self, task: Task, messages: list[dict], tools: list[dict], specs: dict[str, dict], text: str,
              start: float, budget: int) -> str:
        """The output to use for this step: the majority of vote_prompts proposals (text is proposal 0), or text."""
        if self.time_budget_s - (self.clock() - start) < self.vote_prompts * self.min_step_s:
            return text
        apps = set(task.context["installed_apps"])
        texts: list[str | None] = [text]
        for i in range(1, self.vote_prompts):
            picked = self.bank.select(task.request, apps, self.examples, exclude_id=task.id, offset=i * self.examples,
                                      **({"exclude_apps": gold_apps(task.label)} if self.leave_gold_apps_out else {}))
            variant = [system_message(task.context, self.prompt_version, render_examples(picked), self.date_hints), *messages[1:]]
            fits = self.llm.count_tokens(variant, tools) <= budget  # a variant that does not fit gets no vote
            texts.append(self.llm.generate(variant, tools, max_time=self.time_budget_s - (self.clock() - start))
                         if fits else None)
        keyed = [output_key(t, specs) for t in texts]
        keys = [k for k, _ in keyed]
        counts = Counter(k for k, eligible in keyed if eligible)  # only write/ask proposals can win (output_key)
        best = max(counts.values(), default=0)
        top = [k for k, n in counts.items() if n == best]
        winner = keys.index(top[0]) if best >= 2 and len(top) == 1 else 0  # the lowest index with the winning key
        self.last_trace["vote"] = {"step": len(self.last_trace["steps"]), "winner": winner, "applied": self.vote_apply,
                                   "keys": [json.dumps(k, ensure_ascii=False) for k in keys]}
        return texts[winner] if self.vote_apply else text
