import json

import pytest

from lifemate import infer
from lifemate.agents import AGENTS
from lifemate.asr import BACKENDS
from lifemate.data import default_data_root
from lifemate.presets import choose, load_presets, main as presets_main, resolve

needs_data = pytest.mark.skipif(not (default_data_root() / "train").is_dir(), reason="data/train not downloaded")


@pytest.fixture
def tmp_presets(tmp_path):
    path = tmp_path / "presets.json"
    path.write_text(json.dumps({"agent": {"sft-test": {"agent": "react", "agent_kw": {"model": "${HF_MODEL_OWNER}/lifemate-sft-test"}}},
                                "asr": {}}), encoding="utf-8")
    return path


def test_resolve_fills_env_vars(tmp_presets):
    assert resolve("agent", "sft-test", {"HF_MODEL_OWNER": "me"}, tmp_presets) == ("react", {"model": "me/lifemate-sft-test"})


def test_unset_env_var_is_named(tmp_presets):
    with pytest.raises(ValueError, match="HF_MODEL_OWNER"):
        resolve("agent", "sft-test", {}, tmp_presets)


def test_unknown_name_lists_known_names(tmp_presets):
    with pytest.raises(KeyError, match="sft-test"):
        resolve("agent", "nope", {}, tmp_presets)


def test_cached_asr_needs_no_entry():
    assert resolve("asr", "cached:pho-large-fp16") == ("cached", {"name": "pho-large-fp16"})


def test_every_committed_preset_names_a_registered_backend():
    p = load_presets()
    assert all(v["agent"] in AGENTS for v in p["agent"].values())
    assert all(v["asr"] in BACKENDS for v in p["asr"].values())


def test_committed_presets_hold_the_initial_set():
    p = load_presets()
    assert {"empty", "oracle", "react-v1", "react-v2-refs"} <= set(p["agent"])
    assert {"gold", "none", "pho-medium", "pho-large", "pho-large-fp32"} <= set(p["asr"])
    assert resolve("agent", "react-v2-refs") == ("react", {"prompt_version": "v2", "check_refs": True})
    assert resolve("asr", "pho-large-fp32") == ("whisper", {"model": "vinai/PhoWhisper-large", "dtype": "float32"})


def test_list_cli_prints_names_as_json(capsys):
    presets_main(["--list"])
    listed = json.loads(capsys.readouterr().out)
    assert "react-v1" in listed["agent"] and "gold" in listed["asr"]


@needs_data
def test_infer_records_presets(tmp_path):
    out = infer.main(["--split", "train", "--agent-preset", "oracle", "--asr-preset", "gold", "--limit", "2", "--out", str(tmp_path / "r")])
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["agent_preset"] == "oracle" and meta["agent"] == "oracle" and meta["asr_preset"] == "gold"


@needs_data
def test_infer_without_presets_records_null(tmp_path):
    out = infer.main(["--split", "train", "--agent", "empty", "--asr", "none", "--limit", "1", "--out", str(tmp_path / "r")])
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["agent_preset"] is None and meta["asr_preset"] is None


def test_explicit_kw_override_single_preset_keys():
    name, kw = choose("agent", None, "react-v2-refs", ["check_refs=false"], "empty")
    assert (name, kw) == ("react", {"prompt_version": "v2", "check_refs": False})


def test_choose_without_preset_uses_the_plain_backend():
    assert choose("asr", "none", None, ["a=1"], "gold") == ("none", {"a": 1})
    assert choose("asr", None, None, None, "gold") == ("gold", {})


def test_choose_rejects_backend_and_preset_together():
    with pytest.raises(ValueError, match="both"):
        choose("agent", "oracle", "oracle", None, "empty")


def test_agent_and_agent_preset_together_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        infer.main(["--agent", "oracle", "--agent-preset", "oracle", "--out", str(tmp_path / "r")])
    with pytest.raises(SystemExit):
        infer.main(["--asr", "gold", "--asr-preset", "gold", "--out", str(tmp_path / "r")])


def test_infer_passes_the_split_to_agents_that_take_one(tmp_path):
    from lifemate.agents import EmptyAgent, ReActAgent
    root = tmp_path / "data"
    assert infer.with_split(ReActAgent, {"examples": 4}, "public_test", root) == {
        "examples": 4, "split": "public_test", "data_root": root}
    assert infer.with_split(ReActAgent, {"split": "train"}, "public_test", root) == {"split": "train", "data_root": root}
    assert infer.with_split(EmptyAgent, {}, "public_test", root) == {}


def test_fewshot_presets_set_examples():
    assert resolve("agent", "react-fewshot-k2") == ("react", {"examples": 2})
    assert resolve("agent", "react-fewshot-k4") == ("react", {"examples": 4})


class _StubLLM:
    context_limit, max_new_tokens = 16384, 0

    def generate(self, messages, tools, max_time=None):
        return "Xong."

    def count_tokens(self, messages, tools):
        return 1


@needs_data
def test_infer_with_a_react_agent_writes_meta_without_the_data_root(tmp_path, monkeypatch):
    import functools

    from lifemate.agents import ReActAgent
    monkeypatch.setitem(AGENTS, "react", functools.partial(ReActAgent, llm=_StubLLM()))
    out = infer.main(["--split", "train", "--agent", "react", "--asr", "gold", "--limit", "2", "--out", str(tmp_path / "r")])
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["agent_kw"] == {"split": "train"} and meta["agent_info"]["examples"] == 0


@needs_data
def test_infer_builds_the_fewshot_bank_from_the_given_data_root(tmp_path, monkeypatch):
    import functools

    from lifemate.agents import ReActAgent
    monkeypatch.setitem(AGENTS, "react", functools.partial(ReActAgent, llm=_StubLLM()))
    out = infer.main(["--split", "train", "--agent", "react", "--agent-kw", "examples=2", "--asr", "gold", "--limit", "2",
                      "--subset", "dev", "--data-root", str(default_data_root()), "--out", str(tmp_path / "r")])
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["agent_info"]["examples"] == 2 and meta["agent_info"]["bank_size"] == 794
    assert "data_root" not in meta["agent_kw"]


@pytest.mark.parametrize("name", ["sft-v2", "sft-final"])
def test_sft_presets_name_their_merged_model_and_four_examples(name):
    backend, kw = resolve("agent", name, {"HF_MODEL_OWNER": "team"})
    assert backend == "react" and kw["model"] == f"team/lifemate-{name}" and kw["examples"] == 4


def test_fine_tuned_asr_presets_name_their_hub_model():
    for name in ("pho-large-ft-b", "pho-large-ft-all"):
        assert resolve("asr", name, {"HF_MODEL_OWNER": "team"}) == ("whisper", {"model": f"team/{name}"})


def test_check_refs_presets():
    """check_refs passed its gate on sft-v2: holdout +0.60 [+0.03, +1.47], dev +0.0 (all 300 samples unchanged)."""
    env = {"HF_MODEL_OWNER": "team"}
    assert resolve("agent", "sft-v2-refs", env) == (
        "react", {"model": "team/lifemate-sft-v2", "examples": 4, "check_refs": True})
    assert resolve("agent", "sft-final", env) == (
        "react", {"model": "team/lifemate-sft-final", "examples": 4, "check_refs": True})


def test_date_hint_presets_are_their_base_plus_the_flag():
    env = {"HF_MODEL_OWNER": "team"}
    for base, dates in (("sft-v2-refs", "sft-v2-refs-dates"), ("sft-final", "sft-final-dates")):
        name, kw = resolve("agent", base, env)
        assert resolve("agent", dates, env) == (name, {**kw, "date_hints": True})


def test_beam_search_asr_preset_keeps_the_same_model():
    """beam 5: WER 0.0962 -> 0.0911 on the 2108 eval clips, p95 0.66 -> 0.95 s per clip (molab)."""
    env = {"HF_MODEL_OWNER": "team"}
    assert resolve("asr", "pho-large-ft-b-beam5", env) == ("whisper", {"model": "team/pho-large-ft-b", "num_beams": 5})


def test_context_prompt_asr_presets_only_add_the_flag():
    env = {"HF_MODEL_OWNER": "team"}
    for base, context in (("pho-large-ft-b", "pho-large-ft-b-context"), ("pho-large-ft-b-beam5", "pho-large-ft-b-beam5-context")):
        name, kw = resolve("asr", base, env)
        assert resolve("asr", context, env) == (name, {**kw, "context_prompt": True})
