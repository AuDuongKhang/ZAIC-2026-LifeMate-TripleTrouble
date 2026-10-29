from lifemate.agents.fewshot import ExampleBank, render_examples
from lifemate.data import default_data_root
from lifemate.splits import load_splits


def gold_apps(sample):
    return {c["name"].split(".", 1)[0] for c in sample["label"]["tool_calls"]} - {"sys"}


def test_select_never_returns_the_sample_itself(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    s = train_samples[0]
    picked = bank.select(s["audio_text"], set(s["context"]["installed_apps"]), k=3, exclude_id=s["id"])
    assert s["id"] not in [p["id"] for p in picked] and len(picked) == 3


def test_select_only_uses_pool(train_samples):
    pool = {s["id"] for s in train_samples[100:]}
    bank = ExampleBank(train_samples, pool_ids=pool)
    picked = bank.select("chuyển tiền cho mẹ", set(train_samples[0]["context"]["installed_apps"]), k=4, exclude_id=None)
    assert picked and all(p["id"] in pool for p in picked)


def test_select_requires_example_apps_installed(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    picked = bank.select("bật đèn phòng khách", {"smart_home"}, k=2, exclude_id=None)
    assert len(picked) == 2 and all(gold_apps(p) <= {"smart_home"} for p in picked)


def test_select_ranks_the_most_similar_request_first(train_samples):
    bank = ExampleBank(train_samples, pool_ids={s["id"] for s in train_samples})
    s = next(x for x in train_samples if x["label"]["tool_calls"])
    top = bank.select(s["audio_text"], set(s["context"]["installed_apps"]), k=1, exclude_id=None)[0]
    assert top["audio_text"] == s["audio_text"]


def test_render_examples_shows_calls_and_truncated_results():
    ex = [{"id": "x", "audio_text": "tắt đèn",
           "label": {"tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "l1", "power": "off"}}],
                     "tool_results": [{"ok": True, "pad": "z" * 1000}]}}]
    text = render_examples(ex, max_result_chars=50)
    assert "User: tắt đèn" in text and '<tool_call>{"name": "smart_home.light_set"' in text and "z" * 60 not in text


def test_render_examples_marks_out_of_scope_and_empty_selection():
    out_of_scope = [{"id": "y", "audio_text": "kể chuyện cười", "label": {"tool_calls": [], "tool_results": []}}]
    assert "(no tool call)" in render_examples(out_of_scope)
    assert render_examples([]) == ""


def test_from_data_pool_depends_on_the_split(train_samples):
    root = default_data_root()
    assert len(ExampleBank.from_data(root, "train")) == len(load_splits()["pool"])
    assert len(ExampleBank.from_data(root, "public_test")) == len(train_samples) == 1545
