import json
import random

import pytest

from conftest import replay_sample
from lifemate.data import default_data_root, load_catalog
from lifemate.sft import records, remix
from lifemate.sft.mask import _rename_call
from lifemate.splits import load_splits


def _apps():
    root = default_data_root()
    return json.loads((root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))["apps"]


def _table_for(app_ids, suffix="x", params=None):
    """A remix table that renames every tool of the given apps by appending `suffix`, with new descriptions."""
    catalog_apps = {a["id"]: a for a in _apps()}
    table = {"params": dict(params or {}), "apps": {}}
    for app in app_ids:
        spec = catalog_apps[app]
        table["apps"][app] = {"name": f"{app}{suffix}x",
                              "tools": {t["name"].split(".", 1)[1]: {"name": f"{t['name'].split('.', 1)[1]}{suffix}",
                                                                     "description": f"Mô tả mới số {i}."}
                                        for i, t in enumerate(spec["tools"])}}
    return table


@pytest.fixture(scope="module")
def sample_with_two_apps(train_samples):
    pool = set(load_splits()["pool"])
    for s in train_samples:
        calls = [c for c in s["label"]["tool_calls"] if c["name"] != "sys.ask_missing_info"]
        apps = {c["name"].split(".")[0] for c in calls}
        if s["id"] in pool and calls and len(s["context"]["installed_apps"]) >= 3 and len(apps) == 1:
            return s
    pytest.skip("no sample")


def _remixed_apps(sample):
    gold = {c["name"].split(".")[0] for c in sample["label"]["tool_calls"] if c["name"] != "sys.ask_missing_info"}
    return sorted(gold)


# --- the writer and the records refuse holdout apps -------------------------------------------------------------------

def test_remix_refuses_holdout_apps(train_samples, tmp_path):
    holdout = load_splits()["holdout_apps"]
    assert "news" in holdout
    assert not set(remix.apps_to_write(_apps(), holdout)) & set(holdout)
    assert len(remix.apps_to_write(_apps(), holdout)) == len(_apps()) - len(holdout)
    table = {"params": {}, "apps": {"news": {"name": "gazette", "tools": {}}}}
    path = tmp_path / "t.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(SystemExit):
        records.main(["--name", "t", "--subset", "pool", "--limit", "3", "--tokenizer", "none", "--out-dir", str(tmp_path),
                      "--remix", str(path)])


def test_the_table_loader_rejects_holdout_apps():
    with pytest.raises(ValueError, match="news"):
        remix.check_table({"params": {}, "apps": {"news": {"name": "gazette", "tools": {}}}}, load_splits()["holdout_apps"])


# --- remix_record ------------------------------------------------------------------------------------------------------

def _tool_names(rec):
    return [t["function"]["name"] for t in rec["tools"]]


def test_remix_renames_everywhere(sample_with_two_apps):
    s = sample_with_two_apps
    rec = replay_sample(s, examples=4)
    apps = _remixed_apps(s)
    params = sorted({p for t in rec["tools"] if t["function"]["name"].split(".")[0] in apps
                     for p in t["function"]["parameters"].get("properties", {})})
    table = _table_for(apps, params={p: f"{p}_syn" for p in params})
    new = remix.remix_record(rec, table, random.Random(0))
    assert new["source"] == "remixed"
    text = json.dumps({"tools": new["tools"], "messages": new["messages"]}, ensure_ascii=False)
    for app in apps:
        for t in _apps_by_id()[app]["tools"]:
            assert f'"{t["name"]}"' not in text  # no old tool name in schemas or calls
            assert t["name"] not in new["messages"][0]["content"]  # nor in the example calls of the prompt
    names_in_new_tools = set(_tool_names(new))
    assert not (names_in_new_tools & {t["name"] for a in apps for t in _apps_by_id()[a]["tools"]})
    new_apps = {table["apps"][a]["name"] for a in apps}
    for t in new["tools"]:
        if t["function"]["name"].split(".")[0] not in new_apps:
            continue
        for p in params:
            assert p not in t["function"]["parameters"].get("properties", {})  # property keys and required are renamed
            assert p not in t["function"]["parameters"].get("required", [])


def _apps_by_id():
    return {a["id"]: a for a in _apps()}


def test_remix_keeps_absent_apps(sample_with_two_apps):
    s = sample_with_two_apps
    rec = replay_sample(s, examples=4)
    apps = _remixed_apps(s)
    table = _table_for(apps)
    new = remix.remix_record(rec, table, random.Random(0))
    by_name = {t["function"]["name"]: t for t in rec["tools"]}
    absent = [n for n in by_name if n.split(".")[0] not in apps]
    assert absent
    new_by_name = {t["function"]["name"]: t for t in new["tools"]}
    for n in absent:
        assert json.dumps(new_by_name[n], sort_keys=True) == json.dumps(by_name[n], sort_keys=True)


def test_remix_replay_maps_back(sample_with_two_apps):
    s = sample_with_two_apps
    rec = replay_sample(s, examples=4)
    apps = _remixed_apps(s)
    params = sorted({p for t in rec["tools"] if t["function"]["name"].split(".")[0] in apps
                     for p in t["function"]["parameters"].get("properties", {})})
    table = _table_for(apps, params={p: f"{p}_syn" for p in params})
    new = remix.remix_record(rec, table, random.Random(0))
    tool_back = {f"{table['apps'][a]['name']}.{v['name']}": f"{a}.{suffix}"
                 for a in apps for suffix, v in table["apps"][a]["tools"].items()}
    param_back = {v: k for k, v in table["params"].items()}
    new_calls = [m["tool_calls"] for m in new["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
    old_calls = [m["tool_calls"] for m in rec["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(new_calls) == len(old_calls) and old_calls
    for got, want in zip(new_calls, old_calls):
        back = [_rename_call(c["function"]["name"], c["function"]["arguments"], tool_back, param_back) for c in got]
        assert back == [(c["function"]["name"], c["function"]["arguments"]) for c in want]


def test_remix_keeps_the_one_of_note_after_the_new_description():
    rec = {"id": "x", "source": "gold", "messages": [{"role": "system", "content": "s"}],
           "tools": [{"type": "function", "function": {
               "name": "app.do_it", "description": "Cũ. Provide one of: a | b.",
               "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                              "required": []}}}]}
    table = {"params": {"a": "alpha_syn"}, "apps": {"app": {"name": "prog", "tools": {"do_it": {"name": "run_it", "description": "Chạy nó."}}}}}
    new = remix.remix_record(rec, table, random.Random(0))
    fn = new["tools"][0]["function"]
    assert fn["name"] == "prog.run_it"
    assert fn["description"] == "Chạy nó. Provide one of: alpha_syn | b."
    assert set(fn["parameters"]["properties"]) == {"alpha_syn", "b"}


def test_remix_keeps_a_description_word_that_equals_a_parameter_name():
    rec = {"id": "x", "source": "gold", "messages": [{"role": "system", "content": "s"}],
           "tools": [{"type": "function", "function": {
               "name": "app.do_it", "description": "Cũ.",
               "parameters": {"type": "object", "properties": {"date": {"type": "string"}}, "required": []}}}]}
    table = {"params": {"date": "day_syn"}, "apps": {"app": {"name": "prog", "tools": {"do_it": {"name": "run_it", "description": "Đặt date cho việc."}}}}}
    fn = remix.remix_record(rec, table, random.Random(0))["tools"][0]["function"]
    assert fn["description"] == "Đặt date cho việc."  # the new text is not renamed a second time


# --- check_app ---------------------------------------------------------------------------------------------------------

APP = {"id": "app", "tools": [{"name": "app.list_items"}, {"name": "app.add_item"}]}
GOOD = {"name": "shop", "tools": {"list_items": {"name": "show_goods", "description": "Xem hàng."},
                                  "add_item": {"name": "put_goods", "description": "Thêm hàng."}}}


def test_check_app_accepts_a_good_answer():
    assert remix.check_app(APP, GOOD) is None


def test_check_app_rejects_missing_tool():
    bad = {**GOOD, "tools": {"list_items": GOOD["tools"]["list_items"]}}
    assert "missing" in remix.check_app(APP, bad)


@pytest.mark.parametrize("change, word", [
    ({"name": "Bad Name"}, "identifier"),
    ({"name": "app"}, "new"),
])
def test_check_app_rejects_a_bad_app_name(change, word):
    assert word in remix.check_app(APP, {**GOOD, **change})


def test_check_app_rejects_bad_tool_names_and_descriptions():
    tools = lambda **kw: {"tools": {**GOOD["tools"], "add_item": {**GOOD["tools"]["add_item"], **kw}}}  # noqa: E731
    assert "identifier" in remix.check_app(APP, {**GOOD, **tools(name="put-goods")})
    assert "new" in remix.check_app(APP, {**GOOD, **tools(name="add_item")})
    assert "unique" in remix.check_app(APP, {**GOOD, **tools(name="show_goods")})
    assert "description" in remix.check_app(APP, {**GOOD, **tools(description="  ")})
    assert "tool" in remix.check_app(APP, {"name": "shop", "tools": "oops"})
    assert remix.check_app(APP, "oops") is not None


def test_check_params_needs_new_unique_identifiers_that_hit_no_other_name():
    names = {"a", "b", "c"}
    assert remix.check_params({"a": "alpha", "b": "beta"}, names) is None
    assert "identifier" in remix.check_params({"a": "Al pha"}, names)
    assert "new" in remix.check_params({"a": "a"}, names)
    assert "unique" in remix.check_params({"a": "x", "b": "x"}, names)
    assert "collides" in remix.check_params({"a": "c"}, names)  # c exists and is not renamed away


# --- records --remix ---------------------------------------------------------------------------------------------------

def _read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_records_without_remix_unchanged(train_samples, tmp_path):
    a = _read(records.main(["--name", "a", "--subset", "pool", "--limit", "6", "--tokenizer", "none", "--mask", "1",
                            "--out-dir", str(tmp_path)]))
    assert not any(r["source"] == "remixed" for r in a)


def test_records_remix_adds_one_copy_after_the_masked_ones(train_samples, tmp_path, sample_with_two_apps):
    pool_first = next(s for s in train_samples if s["id"] in set(load_splits()["pool"]))
    holdout = set(load_splits()["holdout_apps"])
    table = _table_for([a["id"] for a in _apps() if a["id"] not in holdout])
    path = tmp_path / "t.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    rows = _read(records.main(["--name", "b", "--subset", "pool", "--limit", "4", "--tokenizer", "none", "--mask", "1",
                               "--out-dir", str(tmp_path), "--remix", str(path)]))
    first = [i for i, r in enumerate(rows) if r["id"] == pool_first["id"]]
    assert [rows[i]["source"] for i in first] == ["gold", "masked", "remixed"]
    plain = _read(records.main(["--name", "c", "--subset", "pool", "--limit", "4", "--tokenizer", "none", "--mask", "1",
                                "--out-dir", str(tmp_path)]))
    assert [r for r in rows if r["source"] != "remixed"] == plain  # the other records are untouched


# --- the writer ---------------------------------------------------------------------------------------------------------

class FakeWriter:
    """Answers the writer prompts the way a good teacher would; `bad` apps get no JSON."""

    def __init__(self, bad=()):
        self.bad, self.asked = set(bad), []

    def generate(self, messages):
        first = messages[0]["content"]
        self.asked.append(len(messages))
        if "SPEC: " in first:
            spec = json.loads(first.split("SPEC: ", 1)[1])
            if spec["app"] in self.bad:
                return "không biết"
            return json.dumps({"name": spec["app"] + "_alt", "tools": {t["name"]: {"name": t["name"] + "_alt", "description": "Mô tả khác."}
                                                                       for t in spec["tools"]}})
        names = json.loads(first.split("PARAMS: ", 1)[1])
        return "```json\n" + json.dumps({n: n + "_syn" for n in names}) + "\n```"


def test_writer_skips_holdout_apps_retries_once_and_writes_the_table(tmp_path):
    holdout = load_splits()["holdout_apps"]
    allowed = remix.apps_to_write(_apps(), holdout)
    bad = allowed[0]
    writer = FakeWriter(bad={bad})
    out = remix.main(["--name", "t", "--out-dir", str(tmp_path)], llm=writer)
    table = json.loads(out.read_text(encoding="utf-8"))
    assert set(table["apps"]) == set(allowed) - {bad}
    assert not set(table["apps"]) & set(holdout)
    assert writer.asked.count(3) == 1  # the failed app was asked twice: its retry has the earlier answer and the reason
    assert table["params"] and all(v.endswith("_syn") for v in table["params"].values())
    report = json.loads((tmp_path / "t.report.json").read_text(encoding="utf-8"))
    assert report["apps_passed"] == len(allowed) - 1 and report["failed"].keys() == {bad}
    remix.check_table(table, holdout)  # what the writer wrote is accepted by records --remix
    for app, entry in table["apps"].items():
        spec = next(a for a in _apps() if a["id"] == app)
        assert remix.check_app(spec, entry) is None


def test_clean_params_keeps_only_usable_entries():
    existing = {"a", "b", "c", "d"}
    raw = {"a": "alpha", "b": "alpha", "c": "d", "x": "zed", "d": "Bad Name", "": "q"}
    assert remix.clean_params(raw, existing) == {"a": "alpha"}
    assert remix.clean_params("oops", existing) == {}
