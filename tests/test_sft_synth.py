import copy
import json
from collections import Counter

import pytest

from lifemate.data import default_data_root, import_zaic_tools, load_catalog
from lifemate.evaluate import label_kind
from lifemate.sft import records
from lifemate.sft.synth import env_id, probe, synthetic_sample, verify
from lifemate.splits import load_splits


@pytest.fixture(scope="module")
def parts(train_samples):
    root = default_data_root()
    pool = set(load_splits()["pool"])
    return (import_zaic_tools(root, "train"), load_catalog(root, "train"), root,
            [s for s in train_samples if s["id"] in pool])


def _check(parts, sample, calls, kind):
    zaic_tools, catalog, root, _ = parts
    return verify(sample, calls, kind, catalog, zaic_tools, root)


def _first(parts, kind, pick=lambda s: True):
    return next(s for s in parts[3] if label_kind(s["label"]["tool_calls"]) == kind and pick(s))


def test_verify_accepts_the_gold_calls_of_pool_samples(parts):
    samples = [s for s in parts[3] if label_kind(s["label"]["tool_calls"]) == "act"][:20]
    bad = [(s["id"], _check(parts, s, s["label"]["tool_calls"], "act")) for s in samples]
    assert [b for b in bad if b[1]] == []


def test_verify_accepts_gold_asks_and_empty(parts):
    ask = _first(parts, "ask")
    assert _check(parts, ask, ask["label"]["tool_calls"], "ask") is None
    assert _check(parts, _first(parts, "empty"), [], "empty") is None


def test_verify_rejects_a_bad_enum(parts):
    zaic_tools, catalog, root, _ = parts

    def has_enum(s):
        return any("enum" in catalog[c["name"]]["parameters"].get("properties", {}).get(k, {})
                   for c in s["label"]["tool_calls"] if c["name"] in catalog for k in c["arguments"])

    sample = _first(parts, "act", has_enum)
    calls = copy.deepcopy(sample["label"]["tool_calls"])
    for call in calls:
        props = catalog[call["name"]]["parameters"].get("properties", {})
        for k in call["arguments"]:
            if "enum" in props.get(k, {}):
                call["arguments"][k] = "zzz"
    assert _check(parts, sample, calls, "act").startswith("schema:")


def test_verify_rejects_an_invented_id(parts):
    zaic_tools, catalog, root, _ = parts

    def has_ref(s):
        return any(p.get("x-ref") for c in s["label"]["tool_calls"] if c["name"] in catalog
                   for k, p in catalog[c["name"]]["parameters"].get("properties", {}).items() if k in c["arguments"])

    sample = _first(parts, "act", has_ref)
    calls = copy.deepcopy(sample["label"]["tool_calls"])
    for call in calls:
        props = catalog[call["name"]]["parameters"].get("properties", {})
        for k in call["arguments"]:
            if props.get(k, {}).get("x-ref"):
                call["arguments"][k] = "id_does_not_exist_999"
    assert _check(parts, sample, calls, "act").startswith("unseen id:")


def test_verify_rejects_ask_not_last(parts):
    ask = _first(parts, "ask")
    calls = copy.deepcopy(ask["label"]["tool_calls"])
    second = copy.deepcopy(calls[-1])
    second["arguments"]["params"] = second["arguments"]["params"] + ["extra"]
    assert _check(parts, ask, calls + [second], "ask") == "ask not last"
    assert _check(parts, ask, calls + [calls[0]], "ask") == "kind mismatch"  # last call is not the ask


def test_verify_rejects_calls_on_empty_and_no_calls(parts):
    act = _first(parts, "act")
    assert _check(parts, act, act["label"]["tool_calls"], "empty") == "calls on empty"
    assert _check(parts, act, [], "act") == "no calls"
    assert _check(parts, act, act["label"]["tool_calls"], "ask") == "kind mismatch"


def test_probe_only_calls_reads_without_required_params(parts):
    zaic_tools, catalog, root, pool = parts
    results = probe(pool[0], catalog, zaic_tools, root)
    assert results
    for r in results:
        assert catalog[r["name"]]["kind"] == "read" and not catalog[r["name"]]["parameters"].get("required")
        assert r["arguments"] == {} and r["result"]["ok"]


def test_synthetic_sample_points_at_its_base_environment():
    base = {"id": "train_0001", "audio": "a.wav", "audio_text": "x", "context": {}, "history": [], "label": {}}
    sample = synthetic_sample(base, "xin chào", [], 0)
    assert env_id(sample) == "train_0001" and sample["id"] == "synth_train_0001_0"
    assert env_id(base) == "train_0001"
    assert sample["label"] == {"tool_calls": []} and sample["audio_text"] == "xin chào"


def _write_extra(path, samples):
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")


def _pool_act(train_samples, n=1, offset=0):
    pool = set(load_splits()["pool"])
    return [s for s in train_samples if s["id"] in pool and label_kind(s["label"]["tool_calls"]) == "act"][offset:offset + n]


def _run_records(tmp_path, extra_file, *more):
    return records.main(["--name", "t", "--subset", "pool", "--limit", "1", "--mask", "1", "--examples", "0",
                         "--tokenizer", "none", "--extra-samples", str(extra_file), "--out-dir", str(tmp_path), *more])


def test_records_render_extra_samples(train_samples, tmp_path):
    base = _pool_act(train_samples, 1, offset=3)[0]
    extra = tmp_path / "x.jsonl"
    _write_extra(extra, [synthetic_sample(base, base["audio_text"], base["label"]["tool_calls"], 0)])
    rows = [json.loads(line) for line in _run_records(tmp_path, extra).read_text(encoding="utf-8").splitlines()]
    assert Counter(r["source"] for r in rows) == {"gold": 2, "masked": 2}
    assert sum(r["id"].startswith("synth_") for r in rows) == 2  # the synthetic record and its masked copy


def test_records_refuse_synthetic_samples_on_eval_environments(train_samples, tmp_path):
    dev = set(load_splits()["dev"])
    base = next(s for s in train_samples if s["id"] in dev)
    extra = tmp_path / "x.jsonl"
    _write_extra(extra, [synthetic_sample(base, "x", base["label"]["tool_calls"], 0)])
    with pytest.raises(SystemExit):
        _run_records(tmp_path, extra)


def test_records_skip_the_asr_source_for_extra_samples(train_samples, tmp_path, monkeypatch):
    from test_sft_replay import _write_cache

    monkeypatch.setenv("LIFEMATE_ASR_CACHE", str(tmp_path / "cache"))
    first = _pool_act(train_samples, 1)[0]
    _write_cache(tmp_path, "noisy", [first], lambda clip: "nghe nhầm")
    extra = tmp_path / "x.jsonl"
    _write_extra(extra, [synthetic_sample(first, "yêu cầu mới", first["label"]["tool_calls"], 0)])
    rows = [json.loads(line) for line in _run_records(tmp_path, extra, "--text", "gold", "--text", "cache:noisy"
                                                      ).read_text(encoding="utf-8").splitlines()]
    synth = [r for r in rows if r["id"].startswith("synth_")]
    assert {r["source"] for r in synth} == {"gold", "masked"}
    assert Counter(r["source"] for r in rows if not r["id"].startswith("synth_")) == {"gold": 1, "asr": 1, "masked": 1}


def test_accept_keeps_only_verified_scenarios(parts):
    from lifemate.sft.synth import accept

    zaic_tools, catalog, root, _ = parts
    base = _first(parts, "act")
    gold = base["label"]["tool_calls"]
    scenarios = [{"request": "ok", "kind": "act", "calls": gold},
                 {"request": "bad", "kind": "act", "calls": []},
                 {"request": "empty", "kind": "empty", "calls": []}]
    accepted, reasons, kinds = accept(base, scenarios, catalog, zaic_tools, root)
    assert [s["audio_text"] for s in accepted] == ["ok", "empty"]
    assert [s["id"] for s in accepted] == [f"synth_{base['id']}_0", f"synth_{base['id']}_1"]
    assert dict(reasons) == {"no calls": 1} and dict(kinds) == {"act": 1, "empty": 1}


def test_parse_answer_ignores_fences_and_commentary_and_bad_scenarios():
    from lifemate.sft.synth import parse_answer

    answer = ('Đây là kết quả:\n```json\n{"train_0001": [{"request": " tắt đèn ", "kind": "act", "calls": '
              '[{"name": "smart_home.light_set", "arguments": {"power": "off"}}]}, {"request": "x", "kind": "weird", '
              '"calls": []}, {"request": "y", "kind": "act", "calls": [{"name": "a.b", "arguments": "oops"}]}]}\n```')
    assert parse_answer(answer) == {"train_0001": [{"request": "tắt đèn", "kind": "act", "calls": [
        {"name": "smart_home.light_set", "arguments": {"power": "off"}}]}]}
    assert parse_answer("không có json") == {} and parse_answer("{broken") == {}


def test_pack_and_ingest_roundtrip(train_samples, tmp_path):
    from lifemate.sft import synth

    base = _pool_act(train_samples, 1, offset=2)[0]
    argv = ["--name", "t", "--out-dir", str(tmp_path), "--offset", "2", "--limit", "3", "--per-prompt", "2"]
    synth.main(["pack", *argv])
    prompts = sorted((tmp_path / "t" / "prompts").glob("*.txt"))
    assert len(prompts) == 2
    text = prompts[0].read_text(encoding="utf-8")
    assert "BRIEFS:" in text and '"id"' in text
    answer = {base["id"]: [{"request": "thử", "kind": "act", "calls": base["label"]["tool_calls"]}]}
    (tmp_path / "t" / "scenarios" / "a.txt").write_text("```json\n" + json.dumps(answer) + "\n```", encoding="utf-8")
    synth.main(["ingest", *argv])
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["env_id"] for r in rows] == [base["id"]] and rows[0]["source"] == "synth"


# --- answer: a local teacher writes the scenarios ---------------------------------------------------------------------

class FakeTeacher:
    """Stands in for sft.thoughts.Teacher: generate(messages) -> text."""

    def __init__(self):
        self.messages = []

    def generate(self, messages):
        self.messages.append(messages)
        return f"answer {len(self.messages)}"


def _packed(tmp_path, prompts=3):
    from lifemate.sft import synth

    argv = ["--name", "t", "--out-dir", str(tmp_path), "--limit", str(prompts), "--per-prompt", "1"]
    synth.main(["pack", *argv])
    return synth, ["--name", "t", "--out-dir", str(tmp_path), "--teacher", "x/y", "--revision", "abc"]


def test_answer_writes_one_file_per_prompt(train_samples, tmp_path):
    synth, argv = _packed(tmp_path)
    teacher = FakeTeacher()
    synth.main(["answer", *argv], llm=teacher)
    files = sorted((tmp_path / "t" / "scenarios").glob("prompt_*.json"))
    assert [f.name for f in files] == ["prompt_00.json", "prompt_01.json", "prompt_02.json"]
    assert [f.read_text(encoding="utf-8") for f in files] == ["answer 1", "answer 2", "answer 3"]
    prompts = sorted((tmp_path / "t" / "prompts").glob("prompt_*.txt"))
    assert [m[0]["content"] for m in teacher.messages] == [p.read_text(encoding="utf-8") for p in prompts]
    assert all(m[0]["role"] == "user" and len(m) == 1 for m in teacher.messages)


def test_answer_skips_answered_prompts(train_samples, tmp_path):
    synth, argv = _packed(tmp_path)
    synth.main(["answer", *argv], llm=FakeTeacher())
    again = FakeTeacher()
    synth.main(["answer", *argv], llm=again)
    assert again.messages == []  # a second run makes no LLM call
    (tmp_path / "t" / "scenarios" / "prompt_01.json").unlink()
    third = FakeTeacher()
    synth.main(["answer", *argv], llm=third)
    assert len(third.messages) == 1 and (tmp_path / "t" / "scenarios" / "prompt_01.json").exists()


def test_answer_needs_a_teacher_and_a_revision(train_samples, tmp_path):
    synth, _ = _packed(tmp_path, prompts=1)
    with pytest.raises(SystemExit):
        synth.main(["answer", "--name", "t", "--out-dir", str(tmp_path)])


def test_answer_does_not_import_the_teacher_module_otherwise(train_samples, tmp_path):
    import sys

    sys.modules.pop("lifemate.sft.thoughts", None)
    synth, argv = _packed(tmp_path, prompts=1)
    synth.main(["answer", *argv], llm=FakeTeacher())
    assert "lifemate.sft.thoughts" not in sys.modules


def test_ingest_reads_the_files_answer_wrote(train_samples, tmp_path):
    synth, argv = _packed(tmp_path, prompts=1)
    base = _pool_act(train_samples, 1)[0]
    answer = {base["id"]: [{"request": "thử", "kind": "act", "calls": base["label"]["tool_calls"]}]}

    class Writer(FakeTeacher):
        def generate(self, messages):
            return "```json\n" + json.dumps(answer) + "\n```"

    synth.main(["answer", *argv], llm=Writer())
    synth.main(["ingest", "--name", "t", "--out-dir", str(tmp_path)])
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["env_id"] for r in rows] == [base["id"]]


def test_the_writer_prompt_asks_for_act_ask_pairs_and_stated_arguments_only(train_samples, tmp_path):
    synth, _ = _packed(tmp_path, prompts=1)
    text = (tmp_path / "t" / "prompts" / "prompt_00.txt").read_text(encoding="utf-8")
    assert 'one pair with the same intent' in text
    assert 'The kind "act" one: a detail in the request picks exactly one record' in text
    assert 'The kind "ask" one: several records fit or a required value is missing' in text
    assert "read calls pass only the arguments the user stated" in text.lower()


# --- holdout apps never reach generated data --------------------------------------------------------------------------

def _holdout():
    return set(load_splits()["holdout_apps"])


def _pool_sample_showing_a_holdout_app(parts):
    return next(s for s in parts[3] if set(s["context"]["installed_apps"]) & _holdout())


def _holdout_tool(parts, sample, kind=None):
    catalog = parts[1]
    return next(n for n in (t for app in sample["context"]["installed_apps"].values() for t in app["tools"])
                if n.split(".")[0] in _holdout() and (kind is None or catalog[n]["kind"] == kind))


def test_the_brief_shows_no_holdout_app(parts):
    from lifemate.sft.synth import _brief

    zaic_tools, catalog, root, _ = parts
    sample = _pool_sample_showing_a_holdout_app(parts)
    brief = _brief(sample, catalog, zaic_tools, root)
    assert brief["tools"] and not {n.split(".")[0] for n in brief["tools"]} & _holdout()
    assert not {n.split(".")[0] for n in brief["apps_hold"]} & _holdout()


def test_probe_skips_holdout_apps(parts):
    zaic_tools, catalog, root, _ = parts
    sample = _pool_sample_showing_a_holdout_app(parts)
    assert not {c["name"].split(".")[0] for c in probe(sample, catalog, zaic_tools, root)} & _holdout()


def test_verify_rejects_a_call_to_a_holdout_app(parts):
    sample = _pool_sample_showing_a_holdout_app(parts)
    tool = _holdout_tool(parts, sample)
    assert "holdout" in _check(parts, sample, [{"name": tool, "arguments": {}}], "act")
    ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": tool, "params": ["x"]}}
    assert "holdout" in _check(parts, sample, [ask], "ask")


def test_records_refuse_extra_samples_that_use_a_holdout_app(parts, tmp_path):
    sample = _pool_sample_showing_a_holdout_app(parts)
    tool = _holdout_tool(parts, sample)
    bad = synthetic_sample(sample, "làm đi", [{"name": tool, "arguments": {}}], 0)
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run_records(tmp_path, path)
