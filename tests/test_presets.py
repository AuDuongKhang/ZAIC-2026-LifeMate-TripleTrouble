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
