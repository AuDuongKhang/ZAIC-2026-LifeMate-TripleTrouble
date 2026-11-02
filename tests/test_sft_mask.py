import json
import random
import re

import pytest

from conftest import replay_sample
from lifemate.evaluate import label_kind
from lifemate.sft import records
from lifemate.sft.mask import mask_record
from lifemate.agents.parsing import BLOCK


def _calls(rec):
    return [c["function"] for m in rec["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])]


def _first_record(train_samples, want, examples=0):
    for s in train_samples:
        if want(s):
            rec = replay_sample(s, examples=examples)
            if rec is not None:
                return rec
    raise AssertionError("no sample fits")


@pytest.fixture(scope="module")
def act_record(train_samples):
    """An act sample with several calls and tool calls in its history."""
    return _first_record(train_samples, lambda s: label_kind(s["label"]["tool_calls"]) == "act"
                         and len(s["label"]["tool_calls"]) >= 2
                         and any(t.get("tool_calls") for t in s["history"]))


@pytest.fixture(scope="module")
def ask_record(train_samples):
    return _first_record(train_samples, lambda s: label_kind(s["label"]["tool_calls"]) == "ask")


@pytest.fixture(scope="module")
def examples_record(train_samples):
    return _first_record(train_samples, lambda s: label_kind(s["label"]["tool_calls"]) == "act", examples=4)


def test_mask_renames_consistently(ask_record):
    m = mask_record(ask_record, random.Random(0))
    names = {t["function"]["name"] for t in m["tools"]}
    for msg in m["messages"]:
        for c in msg.get("tool_calls", []):
            assert c["function"]["name"] in names
    ask = m["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert ask["tool_name"] in names
    target = next(t for t in m["tools"] if t["function"]["name"] == ask["tool_name"])
    assert set(ask["params"]) <= set(target["function"]["parameters"]["properties"])


def test_mask_leaves_no_original_names_in_tools(act_record):
    m = mask_record(act_record, random.Random(0))
    text = json.dumps(m["tools"], ensure_ascii=False)
    for t in act_record["tools"]:
        assert t["function"]["name"] not in text or t["function"]["name"] == "sys.ask_missing_info"


def test_mask_keeps_results_and_user_text(act_record):
    m = mask_record(act_record, random.Random(0))
    keep = lambda r: [(x["role"], x["content"]) for x in r["messages"] if x["role"] in ("tool", "user", "system")]
    assert keep(m) == keep(act_record)


def test_mask_is_deterministic(act_record):
    assert mask_record(act_record, random.Random(3)) == mask_record(act_record, random.Random(3))


def test_mask_keeps_arguments_values_and_enums(act_record):
    m = mask_record(act_record, random.Random(0))
    back = {v: k for k, v in m["name_map"].items()}
    assert m["source"] == "masked" and m["id"] == act_record["id"]
    for orig, new in zip(_calls(act_record), _calls(m)):
        assert m["name_map"][new["name"]] == orig["name"] or orig["name"] == new["name"] == "sys.ask_missing_info"
        assert {m["name_map"].get(k, k): v for k, v in new["arguments"].items()} == orig["arguments"]
    enums = lambda r: sorted(json.dumps(p.get("enum")) for t in r["tools"]
                             for p in t["function"]["parameters"]["properties"].values() if "enum" in p)
    assert enums(m) == enums(act_record)


def test_mask_shuffles_the_tool_order(act_record):
    m = mask_record(act_record, random.Random(0))
    order = [m["name_map"].get(t["function"]["name"], t["function"]["name"]) for t in m["tools"]]
    orig = [t["function"]["name"] for t in act_record["tools"]]
    assert sorted(order) == sorted(orig) and len(orig) >= 4 and order != orig


def test_mask_renames_ref_notes_and_required(act_record):
    m = mask_record(act_record, random.Random(0))
    masked_apps = {t["function"]["name"].split(".")[0] for t in m["tools"]} - {"sys"}
    real_apps = {t["function"]["name"].split(".")[0] for t in act_record["tools"]} - {"sys"}
    for t in m["tools"]:
        params = t["function"]["parameters"]
        assert set(params.get("required", [])) <= set(params["properties"])
        for p in params["properties"].values():
            for ref in re.findall(r"id from ([\w.]+)", p.get("description", "")):
                app = ref.split(".")[0]
                assert app in masked_apps or app not in real_apps  # refs to an app not installed stay as they are


def test_mask_renames_the_calls_in_few_shot_examples(examples_record):
    m = mask_record(examples_record, random.Random(0))
    names = {t["function"]["name"] for t in m["tools"]}
    orig_lines = examples_record["messages"][0]["content"].split("\n")
    new_lines = m["messages"][0]["content"].split("\n")
    assert len(orig_lines) == len(new_lines)
    changed = 0
    for old, new in zip(orig_lines, new_lines):
        if not old.startswith("<tool_call>"):
            assert new == old
            continue
        call = json.loads(BLOCK.match(new).group(1))
        assert call["name"] in names
        changed += old != new
    assert changed > 0


def test_masked_records_still_encode(act_record):
    transformers = pytest.importorskip("transformers")
    from lifemate.sft.tokenize import encode_record

    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local HF cache")
    m = mask_record(act_record, random.Random(1))
    enc = encode_record(m, tok)
    labelled = tok.decode([t for t, l in zip(enc["input_ids"], enc["labels"]) if l != -100])
    last_user = max(i for i, x in enumerate(m["messages"]) if x["role"] == "user")
    own = [c["function"]["name"] for x in m["messages"][last_user:] for c in x.get("tool_calls", [])]
    assert own and all(name in labelled for name in own)


def test_records_cli_writes_masked_copies(train_samples, tmp_path):
    out = records.main(["--name", "t", "--subset", "pool", "--limit", "5", "--examples", "0", "--mask", "2",
                        "--tokenizer", "none", "--out-dir", str(tmp_path)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["source"] for r in rows] == ["gold", "masked", "masked"] * 5
    assert rows[1] != rows[2]  # different copies
    again = records.main(["--name", "t2", "--subset", "pool", "--limit", "5", "--examples", "0", "--mask", "2",
                          "--tokenizer", "none", "--out-dir", str(tmp_path)])
    assert again.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")  # same seed, same copies
