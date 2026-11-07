"""Schema paraphrase for SFT: copies of a record in which tools carry other names and other words for the same job.

Function masking (lifemate.sft.mask) swaps names for random words and keeps the description text. A remixed copy gives
the model other wording for the same tools: a local teacher (development only, on a GPU, never at inference) writes, per
app of the train catalog, a new app name and, per tool, a new name and a new Vietnamese description; one more answer
maps parameter names to synonyms for the whole catalog. The result is one file, `train_records/remix/<name>.json`:

    {"params": {old: new}, "apps": {app: {"name": new, "tools": {suffix: {"name": new, "description": text}}}}}

`records --remix FILE` adds one remixed copy of each record after its masked ones. The replay is untouched: a copy renames
the same calls, the example calls in the system prompt and the `sys.ask_missing_info` targets, so the calls stay the
gold calls under other names. An app missing from the table keeps its names and text. Holdout apps are never written and
a table that names one is refused: the gate would no longer be a test on unseen apps.

    python -m lifemate.sft.remix --name r1 --revision <teacher sha> --hf-repo <dataset>        # molab, GPU
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

from ..data import REPO_ROOT, SYS_TOOL, default_data_root
from .mask import _Renamer, _rename_call, _rename_example_calls

REMIX_DIR = REPO_ROOT / "train_records" / "remix"
IDENT = re.compile(r"^[a-z][a-z0-9_]*$")
ONE_OF_NOTE = re.compile(r" Provide one of: .*$")  # the x-one-of note prompting.tool_schema appends
MAX_NEW_TOKENS = 2048
MIN_PASS = 0.8  # share of apps that must pass check_app before the table is worth reading by hand

APP_RULES = """You rename the tools of one app of a Vietnamese voice assistant, for a test of whether a model can use tools
by reading their descriptions instead of remembering their names. You are shown the app (SPEC line). Answer with ONLY one
JSON object:

{"name": "<new app name>", "tools": {"<old tool name>": {"name": "<new tool name>", "description": "<new description>"}}}

Rules:
- Every tool of the app appears once, under its old name. The new app name and the new tool names are English
  snake_case identifiers (lowercase letters, digits and underscores, starting with a letter), all different from the old
  ones, and the new tool names are different from each other. Do not reuse the name of another app.
- The new description is Vietnamese, says the same thing in other words, and keeps what the old one says about which
  arguments to use and what the tool returns. Do not change what a tool does. At most 30 words.
- Do not write parameter names, ids or example values that are not in the old description.
"""
PARAM_RULES = """You give synonyms for the parameter names of a voice assistant's tools. You are shown the names (PARAMS line).
Answer with ONLY one JSON object mapping each old name to a new one:

{"<old name>": "<new name>"}

Rules: each new name is an English snake_case identifier (lowercase letters, digits and underscores, starting with a
letter) that means the same, different from the old one and from every other new name, and from every old name that
stays. A name that ends in "_id" keeps that ending. Skip a name rather than change its meaning.
"""
RETRY = "That answer was rejected: {reason}. Write the JSON object again so that it follows every rule. Reply with the JSON only."


def apps_to_write(apps: list[dict], holdout: list[str]) -> list[str]:
    """Ids of the catalog apps the writer may paraphrase: every app but the holdout ones."""
    return [a["id"] for a in apps if a["id"] not in set(holdout)]


def _ident_error(label: str, name) -> str | None:
    if not isinstance(name, str) or not IDENT.match(name):
        return f"{label} {name!r} is not an identifier"
    return None


def check_app(app_spec: dict, answer, taken=frozenset()) -> str | None:
    """None when `answer` is a usable paraphrase of the catalog app `app_spec`, else the reason: every tool present,
    identifiers valid, the app name and the tool names new, the tool names unique, descriptions not empty. `taken` holds
    names the new app name must not equal: every catalog app id (holdout ones included) and the names given so far."""
    if not isinstance(answer, dict):
        return "the answer is not a JSON object"
    reason = _ident_error("app name", answer.get("name"))
    if reason:
        return reason
    if answer["name"] == app_spec["id"]:
        return f"app name {answer['name']!r} is not new"
    if answer["name"] in taken:
        return f"app name {answer['name']!r} is taken by another app"
    tools = answer.get("tools")
    if not isinstance(tools, dict):
        return "tools is not an object"
    new_names = []
    for tool in app_spec["tools"]:
        suffix = tool["name"].split(".", 1)[1]
        entry = tools.get(suffix)
        if entry is None:
            return f"tool {suffix} is missing"
        if not isinstance(entry, dict):
            return f"tool {suffix} is not an object"
        reason = _ident_error(f"tool {suffix}: name", entry.get("name"))
        if reason:
            return reason
        if entry["name"] == suffix:
            return f"tool {suffix}: name {entry['name']!r} is not new"
        description = entry.get("description")
        if not isinstance(description, str) or not description.strip():
            return f"tool {suffix}: description is empty"
        new_names.append(entry["name"])
    if len(set(new_names)) != len(new_names):
        return "new tool names are not unique"
    return None


def check_params(params: dict, existing: set[str]) -> str | None:
    """None when the synonym table is usable: new identifiers, each different from its old name, unique, and not equal
    to an existing parameter name that is not itself renamed."""
    if not isinstance(params, dict):
        return "params is not an object"
    for old, new in params.items():
        reason = _ident_error(f"parameter {old}: name", new)
        if reason:
            return reason
        if new == old:
            return f"parameter {old}: name {new!r} is not new"
    values = list(params.values())
    if len(set(values)) != len(values):
        return "new parameter names are not unique"
    for old, new in params.items():
        if new in existing and new not in params:
            return f"parameter {old}: name {new!r} collides with an existing parameter"
    return None


def clean_params(raw, existing: set[str]) -> dict[str, str]:
    """The entries of a teacher's answer that can stay, in order: known old names, valid identifiers, each new name new,
    unused and not an existing name."""
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    used = set()
    for old, new in raw.items():
        if old in existing and isinstance(new, str) and IDENT.match(new) and new != old and new not in used \
                and new not in existing:
            out[old] = new
            used.add(new)
    return out


def check_table(table: dict, holdout: list[str], known: set[str] | None = None) -> None:
    """Raises ValueError when the table names a holdout app, is not shaped like a remix table, gives two apps one new
    name, or (with `known`, the catalog's app ids) gives an app the name of another existing app."""
    if not isinstance(table, dict) or not isinstance(table.get("apps"), dict) or not isinstance(table.get("params"), dict):
        raise ValueError("a remix table is {'params': {...}, 'apps': {...}}")
    named = sorted(set(table["apps"]) & set(holdout))
    if named:
        raise ValueError(f"the remix table names holdout apps: {', '.join(named)}")
    new = [entry["name"] for entry in table["apps"].values()]
    repeated = sorted({n for n in new if new.count(n) > 1} | (set(new) & (known or set())))
    if repeated:
        raise ValueError(f"new app names are not unique or are existing app names: {', '.join(repeated)}")


def applies(record: dict, table: dict) -> bool:
    """The record shows at least one tool of an app the table rewrites."""
    return any(_target(t["function"]["name"], table) for t in record["tools"])


def _target(name: str, table: dict) -> str | None:
    app, _, suffix = name.partition(".")
    entry = table["apps"].get(app)
    if entry and suffix in entry["tools"]:
        return f"{entry['name']}.{entry['tools'][suffix]['name']}"
    return None


def remix_record(record: dict, table: dict, rng: random.Random) -> dict:
    """A copy of an SFT record with source="remixed": the tools of the apps in the table renamed and re-described, their
    parameter names replaced by the table's synonyms, every call and example call renamed to match, tools shuffled. An
    app missing from the table keeps its names and text. The input is not modified."""
    tool_map = {t["function"]["name"]: new for t in record["tools"] if (new := _target(t["function"]["name"], table))}
    app_map = {name.split(".")[0]: new.split(".")[0] for name, new in tool_map.items()}
    own_params = {p for t in record["tools"] if t["function"]["name"] in tool_map
                  for p in t["function"]["parameters"].get("properties", {})}
    param_map = {p: table["params"].get(p, p) for p in own_params}  # identity for a name the table leaves alone
    renamer = _Renamer(tool_map, app_map, param_map)

    new_tools = []
    for t in record["tools"]:
        fn = t["function"]
        if fn["name"] == SYS_TOOL:
            new_tools.append(renamer.descriptions(copy.deepcopy(t), []))  # its parameters keep their names
            continue
        if fn["name"] not in tool_map:
            new_tools.append(copy.deepcopy(t))  # an app the table does not rewrite: byte-identical
            continue
        schema = dict(fn["parameters"])
        schema["properties"] = {param_map.get(k, k): v for k, v in fn["parameters"].get("properties", {}).items()}
        if "required" in schema:
            schema["required"] = [param_map.get(p, p) for p in schema["required"]]
        own = list(fn["parameters"].get("properties", {}))
        renamed = renamer.descriptions(copy.deepcopy(dict(t, function=dict(fn, parameters=schema))), own)
        note = ONE_OF_NOTE.search(renamed["function"]["description"])  # renamed with the parameters, kept at the end
        app, _, suffix = fn["name"].partition(".")
        entry = table["apps"][app]["tools"][suffix]
        renamed["function"]["name"] = tool_map[fn["name"]]
        renamed["function"]["description"] = entry["description"].strip() + (note.group(0) if note else "")
        new_tools.append(renamed)
    names = [t["function"]["name"] for t in new_tools]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f"{record.get('id')}: the remixed copy would hold two tools named {', '.join(duplicated)}")
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
    return dict(record, messages=messages, tools=new_tools, source="remixed")


def parse_json(text: str):
    """The first JSON object in a chat answer (fences and commentary around it are ignored), or None."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return obj


def app_prompt(app_spec: dict) -> str:
    spec = {"app": app_spec["id"], "name_vi": app_spec.get("name_vi"), "description": app_spec.get("description"),
            "tools": [{"name": t["name"].split(".", 1)[1], "kind": t.get("kind"), "description": t.get("description"),
                       "parameters": list((t.get("parameters") or {}).get("properties", {})), "returns": t.get("returns")}
                      for t in app_spec["tools"]]}
    return APP_RULES + "\nSPEC: " + json.dumps(spec, ensure_ascii=False)


def params_prompt(names: list[str]) -> str:
    return PARAM_RULES + "\nPARAMS: " + json.dumps(sorted(names), ensure_ascii=False)


def _ask(llm, prompt: str, check):
    """(answer, reason): the answer to `prompt`, once more with the reason when check() rejects the first one."""
    messages = [{"role": "user", "content": prompt}]
    text = llm.generate(messages)
    answer = parse_json(text)
    reason = check(answer) if answer is not None else "the answer holds no JSON object"
    if reason:
        messages += [{"role": "assistant", "content": text}, {"role": "user", "content": RETRY.format(reason=reason)}]
        text = llm.generate(messages)
        answer = parse_json(text)
        reason = check(answer) if answer is not None else "the answer holds no JSON object"
    return (None if reason else answer), reason


def main(argv: list[str] | None = None, llm=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--revision", required=llm is None, help="teacher commit sha, the one used for the reasoning sentences")
    ap.add_argument("--hf-repo", default=None, help="private HF dataset; uploads remix/<name>.json and the report")
    ap.add_argument("--data-root", type=Path, default=default_data_root())
    ap.add_argument("--out-dir", type=Path, default=REMIX_DIR)
    args = ap.parse_args(argv)
    from ..splits import load_splits

    catalog = json.loads((args.data_root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))["apps"]
    holdout = load_splits()["holdout_apps"]
    wanted = set(apps_to_write(catalog, holdout))
    apps = [a for a in catalog if a["id"] in wanted and a["tools"]]
    names = {p for a in apps for t in a["tools"] for p in (t.get("parameters") or {}).get("properties", {})}
    if llm is None:
        from .thoughts import Teacher

        llm = Teacher(args.teacher, args.revision, max_new_tokens=MAX_NEW_TOKENS)

    table: dict = {"params": {}, "apps": {}}
    report: dict = {"teacher": args.teacher, "revision": args.revision, "apps_asked": len(apps), "failed": {}}
    taken = {a["id"] for a in catalog}  # every existing app, holdout ones included, and the new names given so far
    for i, app in enumerate(apps):
        answer, reason = _ask(llm, app_prompt(app), lambda a, app=app: check_app(app, a, taken))
        if answer is None:
            report["failed"][app["id"]] = reason  # left out after one retry
        else:
            taken.add(answer["name"])
            table["apps"][app["id"]] = {"name": answer["name"], "tools": {
                t["name"].split(".", 1)[1]: {"name": answer["tools"][t["name"].split(".", 1)[1]]["name"],
                                             "description": answer["tools"][t["name"].split(".", 1)[1]]["description"].strip()}
                for t in app["tools"]}}
        print(f"app {app['id']} ({i + 1}/{len(apps)}): {'ok' if answer else 'failed: ' + str(reason)}", flush=True)
    raw, _ = _ask(llm, params_prompt(sorted(names)), lambda a: None if isinstance(a, dict) else "not an object")
    table["params"] = clean_params(raw, names)
    reason = check_params(table["params"], names)
    if reason:  # clean_params keeps only valid entries, so this is a bug guard
        raise SystemExit(f"the parameter table is invalid: {reason}")
    check_table(table, holdout, {a["id"] for a in catalog})
    report.update(apps_passed=len(table["apps"]), share_passed=round(len(table["apps"]) / max(len(apps), 1), 4),
                  params_renamed=len(table["params"]), params_total=len(names))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / f"{args.name}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}: {report}", flush=True)
    if report["share_passed"] < MIN_PASS:
        print(f"WARNING: only {report['share_passed']:.0%} of the apps passed check_app (the plan stops below "
              f"{MIN_PASS:.0%}): report to K before training", flush=True)
    if args.hf_repo:
        from huggingface_hub import HfApi

        for path in (out, args.out_dir / f"{args.name}.report.json"):
            HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"remix/{path.name}", repo_id=args.hf_repo,
                                repo_type="dataset")
        print(f"uploaded remix/{args.name}.json to {args.hf_repo}", flush=True)
    return out


if __name__ == "__main__":
    main()
