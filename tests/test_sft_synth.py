import copy
import json
from collections import Counter

import pytest

from lifemate.data import default_data_root, import_zaic_tools, load_catalog
from lifemate.evaluate import label_kind
from lifemate.sft import records
from lifemate.sft.synth import env_id, focus_bases, main, probe, shape, synthetic_sample, uses_app, verify
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


def test_focus_bases_caps_per_app_and_uses_a_base_once():
    def base(i, *apps):
        return {"id": i, "context": {"installed_apps": list(apps)}}
    bases = [base("a", "news", "loyalty"), base("b", "news"), base("c", "news", "loyalty"), base("d", "other")]
    assert focus_bases(bases, ["news", "loyalty"], 1) == {"a": "loyalty", "b": "news"}
    assert focus_bases(bases, ["news", "loyalty"], 9) == {"a": "loyalty", "c": "loyalty", "b": "news"}


def test_uses_app_counts_the_app_an_ask_is_about():
    ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "news.search", "params": ["query"]}}
    assert uses_app([ask], "news") and not uses_app([ask], "loyalty")
    assert uses_app([{"name": "x.y", "arguments": {}}, {"name": "news.list", "arguments": {}}], "news")
    assert not uses_app([], "news")


def test_pack_with_apps_targets_pool_samples_with_that_app_and_ingest_rejects_others(train_samples, tmp_path):
    main(["pack", "--name", "f", "--apps", "holdout", "--limit", "2", "--per-prompt", "50", "--out-dir", str(tmp_path)])
    run = tmp_path / "f"
    focus = json.loads((run / "focus.json").read_text(encoding="utf-8"))
    held, pool = load_splits()["holdout_apps"], set(load_splits()["pool"])
    assert Counter(focus.values()) == {app: 2 for app in held} and set(focus) <= pool
    by_id = {s["id"]: s for s in train_samples}
    assert all(app in by_id[i]["context"]["installed_apps"] for i, app in focus.items())
    text = (run / "prompts" / "prompt_00.txt").read_text(encoding="utf-8")
    assert "contrast pairs" in text and "Read calls pass only the arguments the user stated" in text
    assert "An ask normally comes after reads" in text and "usually called with {}" in text
    assert all(f'"focus_app": "{app}"' in text for app in held)
    assert all(by_id[i]["audio_text"] not in text for i in focus)  # the base's own request stays out

    base_id, app = next(iter(focus.items()))
    answer = {base_id: [{"request": "việc khác", "kind": "empty", "calls": []},
                        {"request": "hỏi", "kind": "ask", "calls": [
                            {"name": "sys.ask_missing_info", "arguments": {"tool_name": f"{app}.x", "params": ["y"]}}]}]}
    (run / "scenarios" / "a.json").write_text(json.dumps(answer), encoding="utf-8")
    main(["ingest", "--name", "f", "--out-dir", str(tmp_path)])
    report = json.loads((tmp_path / "f.report.json").read_text(encoding="utf-8"))
    assert report["rejected_by_reason"]["off focus"] == 1 and report["scenarios"] == 2


def test_pack_without_apps_keeps_the_general_prompt(train_samples, tmp_path):
    main(["pack", "--name", "g", "--limit", "1", "--out-dir", str(tmp_path)])
    text = (tmp_path / "g" / "prompts" / "prompt_00.txt").read_text(encoding="utf-8")
    assert "contrast pairs" not in text and "focus_app" not in text and not (tmp_path / "g" / "focus.json").exists()


def test_shape_separates_direct_asks_from_asks_after_reads():
    catalog = {"a.list": {"kind": "read"}, "a.do": {"kind": "write"}}
    def sample(text, *calls):
        return {"audio_text": text, "label": {"tool_calls": [{"name": n, "arguments": args} for n, args in calls]}}
    ask = ("sys.ask_missing_info", {"tool_name": "a.do", "params": ["x"]})
    got = shape([sample("hai từ", ask), sample("một hai ba bốn", ("a.list", {}), ask),
                 sample("một", ("a.list", {"q": "y"}), ("a.do", {})), sample("một", ("a.list", {}))], catalog)
    assert got["asks_after_a_read"] == 0.5 and got["acts_with_a_write"] == 0.5 and got["calls_per_act"] == 1.5
    assert got["words_per_ask"] == 3.0 and got["words_per_act"] == 1.0 and got["reads_without_args"] == 0.67
    assert shape([], catalog)["asks_after_a_read"] == 0.0
