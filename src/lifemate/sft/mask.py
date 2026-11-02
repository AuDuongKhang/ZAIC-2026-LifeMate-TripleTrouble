"""Function masking (Hammer, Lin et al. 2025): copies of an SFT record in which app prefixes, tool name suffixes and
parameter names are replaced by random but plausible identifiers and the tool order is shuffled, so the model has to
read the descriptions instead of relying on names it memorised. Unseen apps in the test splits are the target.

The same renaming is applied everywhere a name occurs: tool schemas (names, property keys, `required`), every
assistant tool call (history, the agent's own steps, the few-shot example calls in the system prompt), the
`tool_name`/`params` values of `sys.ask_missing_info`, and whole-word occurrences inside descriptions (the
`Provide one of: a + b` suffix, mentions in the text; the `(id from <app>.<collection>)` notes get only
the app prefix renamed). Everything else is kept: other
description text, enum values, argument values, user text, tool results. Only top-level parameters are renamed; keys
inside array items (food_delivery.place_order `items`) stay.
"""
from __future__ import annotations

import copy
import json
import random
import re

from ..agents.parsing import BLOCK
from ..data import SYS_TOOL

APP_WORDS = """atlas beacon bento canvas cascade cedar comet compass coral crest delta ember falcon fern fjord forge
galaxy garnet glacier harbor hazel horizon indigo iris jade juniper kestrel lagoon lantern lotus maple meadow mosaic
nebula nimbus oasis onyx orbit pebble pine prism quartz quill raven reef ridge river saffron sage sequoia sierra
sparrow spruce summit tango thistle tundra umber valley velvet willow zephyr""".split()

VERB_WORDS = """add adjust apply archive assign attach begin bind browse cancel capture change check clear close
collect commit compose compute confirm count create define delete describe detach dispatch drop edit enable end
enroll erase fetch filter find finish grant halt hold inspect issue join launch list load locate log lookup make mark
modify move note obtain open order pause place post query read record redeem refresh register release remove rename
renew request reserve reset resolve resume retrieve review revoke run save scan schedule seek select send set show
start stop store submit suspend switch sync toggle track transfer trigger undo unlock update upload view watch""".split()

PARAM_WORDS = """alpha anchor angle area badge band base batch beam block bound bucket bundle cap card cell chain
chunk clip code core count cue degree depth detail digit draft entry factor field figure flag focus frame gauge grade
grid group handle hint index input item key kind layer level limit line link mark measure memo mode node notch offset
option origin pack pair panel part path patch phase piece pin pitch point portion post quota range rank rate ratio
record ref region role rule scale scope score segment shape signal size slot source span spec stage stamp state step
tag target term text tier title token topic total trace track unit value variant vector weight window
zone""".split()


def _identifiers(originals: list[str], words: list[str], rng: random.Random, taken: set[str],
                 keep_id_suffix: bool = False) -> dict[str, str]:
    """{original: new identifier}, one-to-one, never an identifier in taken. Two words joined by "_" once the word
    list runs out. With keep_id_suffix, "<x>_id" becomes "<word>_id" (it still reads as an id)."""
    out: dict[str, str] = {}
    pool = words[:]
    rng.shuffle(pool)
    for orig in originals:
        while True:
            word = pool.pop() if pool else f"{rng.choice(words)}_{rng.choice(words)}"
            new = f"{word}_id" if keep_id_suffix and orig.endswith("_id") else word
            if new not in taken:
                break
        taken.add(new)
        out[orig] = new
    return out


class _Renamer:
    """Whole-word renaming inside description text: tool names; other dotted names of a renamed app, such as the
    `(id from music.playlists)` notes, get only their app prefix renamed (the rest is a hint, like a description);
    parameter names that are not part of a dotted name."""

    def __init__(self, tool_map: dict[str, str], app_map: dict[str, str], param_map: dict[str, str]):
        self.tool_map, self.app_map, self.param_map = tool_map, app_map, param_map
        alts = [rf"(?<![\w.]){re.escape(n)}(?!\w)" for n in sorted(tool_map, key=len, reverse=True)]
        if app_map:
            apps = "|".join(re.escape(a) for a in sorted(app_map, key=len, reverse=True))
            alts.append(rf"(?<![\w.])(?:{apps})\.\w+")
        alts += [rf"(?<![\w.]){re.escape(n)}(?!\w|\.\w)" for n in sorted(param_map, key=len, reverse=True)]
        self.pattern = re.compile("|".join(alts)) if alts else None

    def _one(self, m: re.Match) -> str:
        word = m.group(0)
        if word in self.tool_map:
            return self.tool_map[word]
        if word in self.param_map:
            return self.param_map[word]
        app, _, rest = word.partition(".")
        return f"{self.app_map[app]}.{rest}"

    def text(self, text: str) -> str:
        return self.pattern.sub(self._one, text) if self.pattern else text

    def descriptions(self, node):
        if isinstance(node, dict):
            return {k: (self.text(v) if k == "description" and isinstance(v, str) else self.descriptions(v))
                    for k, v in node.items()}
        if isinstance(node, list):
            return [self.descriptions(v) for v in node]
        return node


def _rename_call(name: str, arguments: dict, tool_map: dict[str, str], param_map: dict[str, str]) -> tuple[str, dict]:
    if name == SYS_TOOL:
        target = arguments.get("tool_name")
        if target not in tool_map:
            return name, arguments
        args = dict(arguments, tool_name=tool_map[target])
        if isinstance(args.get("params"), list):
            args["params"] = [param_map.get(p, p) for p in args["params"]]
        return name, args
    if name not in tool_map:  # a tool this record does not show: nothing to stay consistent with
        return name, arguments
    return tool_map[name], {param_map.get(k, k): v for k, v in arguments.items()}


def _rename_example_calls(system: str, tool_map, param_map) -> str:
    """The few-shot examples' `<tool_call>{...}</tool_call>` lines in the system prompt; every other line is kept."""
    lines = system.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("<tool_call>"):
            continue
        m = BLOCK.fullmatch(line)
        try:
            call = json.loads(m.group(1)) if m else None
        except json.JSONDecodeError:
            call = None
        if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
            continue
        name, args = _rename_call(call["name"], call["arguments"], tool_map, param_map)
        lines[i] = "<tool_call>" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "</tool_call>"
    return "\n".join(lines)


def mask_record(record: dict, rng: random.Random) -> dict:
    """A masked copy of an SFT record ({"id", "messages", "tools", ...}) with source="masked" and name_map
    {new name: original name} for the tool and parameter names. The input is not modified."""
    tools = [t for t in record["tools"] if t["function"]["name"] != SYS_TOOL]
    names = sorted(t["function"]["name"] for t in tools)
    apps = sorted({n.split(".", 1)[0] for n in names})
    suffixes = sorted({n.split(".", 1)[1] for n in names if "." in n})
    params = sorted({p for t in tools for p in t["function"]["parameters"].get("properties", {})})
    taken = set(apps) | set(suffixes) | set(params) | set(names) | {"sys", "ask_missing_info", "tool_name", "params"}
    app_map = _identifiers(apps, APP_WORDS, rng, taken)
    suffix_map = _identifiers(suffixes, VERB_WORDS, rng, taken)
    param_map = _identifiers(params, PARAM_WORDS, rng, taken, keep_id_suffix=True)
    tool_map = {}
    for n in names:
        app, _, suffix = n.partition(".")
        tool_map[n] = f"{app_map[app]}.{suffix_map[suffix]}" if suffix else app_map[app]
    renamer = _Renamer(tool_map, app_map, param_map)

    new_tools = []
    for t in record["tools"]:
        fn = t["function"]
        if fn["name"] == SYS_TOOL:
            new_tools.append(renamer.descriptions(copy.deepcopy(t)))
            continue
        params_schema = dict(fn["parameters"])
        params_schema["properties"] = {param_map[k]: v for k, v in fn["parameters"].get("properties", {}).items()}
        if "required" in params_schema:
            params_schema["required"] = [param_map.get(p, p) for p in params_schema["required"]]
        new_fn = dict(fn, name=tool_map[fn["name"]], parameters=params_schema)
        new_tools.append(renamer.descriptions(copy.deepcopy(dict(t, function=new_fn))))
    rng.shuffle(new_tools)

    messages = []
    for msg in record["messages"]:
        msg = copy.deepcopy(msg)
        if msg["role"] == "system":
            msg["content"] = _rename_example_calls(msg["content"], tool_map, param_map)
        for call in msg.get("tool_calls", []):
            fn = call["function"]
            fn["name"], fn["arguments"] = _rename_call(fn["name"], fn["arguments"], tool_map, param_map)
        messages.append(msg)

    name_map = {new: old for old, new in tool_map.items()} | {new: old for old, new in param_map.items()}
    return dict(record, messages=messages, tools=new_tools, source="masked", name_map=name_map)
