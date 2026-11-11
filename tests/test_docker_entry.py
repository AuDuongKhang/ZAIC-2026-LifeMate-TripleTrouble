import json
import os

import pytest

from lifemate import docker_entry


def _split_dir(tmp_path, n_samples=1, name="private_test"):
    d = tmp_path / "mounted" / name  # the organisers mount the folder that holds the split folder at /data
    d.mkdir(parents=True)
    (d / f"{name}_tools_catalog.json").write_text("{}", encoding="utf-8")
    (d / f"{name}.jsonl").write_text("".join(json.dumps({"id": f"p{i}"}) + "\n" for i in range(n_samples)),
                                     encoding="utf-8")
    return d


def _train_dir(tmp_path):
    t = tmp_path / "train_src"
    t.mkdir()
    (t / "train.jsonl").write_text('{"id": "train_0001"}\n', encoding="utf-8")
    (t / "train_tools_catalog.json").write_text("{}", encoding="utf-8")
    return t


def _config(tmp_path):
    cfg = {"agent_preset": "empty", "asr_preset": "none", "agent_kw": {"revision": "1e5"}, "asr_kw": {},
           "hf_model_owner": "team"}
    p = tmp_path / "shipped.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _fake_infer(tmp_path, lines, record):
    def fake_main(argv):
        record["argv"] = argv
        record["env"] = dict(os.environ)
        record["data_root"] = argv[argv.index("--data-root") + 1]
        run = tmp_path / "run"
        run.mkdir(exist_ok=True)
        (run / "submission.jsonl").write_text("".join(lines), encoding="utf-8")
        return run
    return fake_main


def _run(tmp_path, monkeypatch, lines, n_samples=1):
    import lifemate.infer

    record = {}
    monkeypatch.setattr(lifemate.infer, "main", _fake_infer(tmp_path, lines, record))
    out = tmp_path / "result" / "submission.jsonl"
    argv = ["--data", str(_split_dir(tmp_path, n_samples).parent), "--out", str(out), "--config", str(_config(tmp_path)),
            "--work", str(tmp_path / "work"), "--train-dir", str(_train_dir(tmp_path))]
    try:
        docker_entry.main(argv)
    except OSError as e:  # symlinks need developer mode on Windows
        pytest.skip(f"symlinks not available: {e}")
    return record, out


def test_find_split_reads_the_catalog_name(tmp_path):
    assert docker_entry.find_split(_split_dir(tmp_path)) == "private_test"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="tools_catalog"):
        docker_entry.find_split(empty)


def test_find_split_refuses_two_catalogs(tmp_path):
    d = _split_dir(tmp_path)
    (d / "other_tools_catalog.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="other_tools_catalog.json"):
        docker_entry.find_split(d)


def test_entry_sets_offline_flags_before_infer(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    record, out = _run(tmp_path, monkeypatch, ['{"id": "p0", "tool_calls": [], "state_hash": "h"}\n'])
    assert record["env"]["HF_HUB_OFFLINE"] == "1" and record["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert record["env"]["HF_MODEL_OWNER"] == "team"
    assert out.read_text(encoding="utf-8") == '{"id": "p0", "tool_calls": [], "state_hash": "h"}\n'


def test_entry_lays_out_the_split_and_the_train_bank(tmp_path, monkeypatch):
    record, _ = _run(tmp_path, monkeypatch, ['{"id": "p0", "tool_calls": [], "state_hash": "h"}\n'])
    root = record["data_root"]
    assert os.path.isfile(os.path.join(root, "private_test", "private_test.jsonl"))
    assert os.path.isfile(os.path.join(root, "train", "train.jsonl"))
    assert os.path.isfile(os.path.join(root, "train", "train_tools_catalog.json"))
    argv = record["argv"]
    assert argv[argv.index("--split") + 1] == "private_test"
    assert argv[argv.index("--agent-preset") + 1] == "empty" and argv[argv.index("--asr-preset") + 1] == "none"


def test_entry_asks_infer_for_the_time_field(tmp_path, monkeypatch):
    record, _ = _run(tmp_path, monkeypatch, ['{"id": "p0", "tool_calls": [], "state_hash": "h", "time": 3}\n'])
    assert "--time-field" in record["argv"]


def test_entry_passes_config_kwargs_as_json(tmp_path, monkeypatch):
    """A revision such as "1e5" must stay a string: infer parses KEY=VALUE values as JSON."""
    record, _ = _run(tmp_path, monkeypatch, ['{"id": "p0", "tool_calls": [], "state_hash": "h"}\n'])
    argv = record["argv"]
    kw = argv[argv.index("--agent-kw") + 1]
    key, _, value = kw.partition("=")
    assert key == "revision" and json.loads(value) == "1e5"


def test_entry_fails_on_a_short_submission(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, monkeypatch, [], n_samples=1)
    assert e.value.code not in (0, None)


def _fetched(monkeypatch, cfg):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo_id, revision=None, **kw: calls.append((repo_id, revision)))
    docker_entry.fetch_models(cfg)
    return calls


def test_fetch_models_downloads_both_models_at_their_revisions(monkeypatch):
    cfg = {"agent_preset": "sft-final", "asr_preset": "pho-large-ft-b", "agent_kw": {"revision": "r1"},
           "asr_kw": {"revision": "r2"}, "hf_model_owner": "team"}
    assert _fetched(monkeypatch, cfg) == [("team/lifemate-sft-final", "r1"), ("team/pho-large-ft-b", "r2")]


def test_fetch_models_uses_the_backend_default_model(monkeypatch):
    cfg = {"agent_preset": "react-v1", "asr_preset": "none", "agent_kw": {}, "asr_kw": {}, "hf_model_owner": "team"}
    assert _fetched(monkeypatch, cfg) == [("Qwen/Qwen3-4B-Instruct-2507", None)]


def test_fetch_models_skips_backends_without_a_model(monkeypatch):
    cfg = {"agent_preset": "empty", "asr_preset": "none", "agent_kw": {}, "asr_kw": {}, "hf_model_owner": "team"}
    assert _fetched(monkeypatch, cfg) == []


def test_committed_shipped_config_names_known_presets():
    from lifemate.presets import resolve

    cfg = docker_entry.load_config()
    env = {"HF_MODEL_OWNER": cfg["hf_model_owner"]}
    resolve("agent", cfg["agent_preset"], env)
    resolve("asr", cfg["asr_preset"], env)


def test_locate_split_accepts_the_parent_folder(tmp_path):
    d = _split_dir(tmp_path)
    assert docker_entry.locate_split(d.parent) == (d.parent / "private_test", "private_test")


def test_locate_split_accepts_the_split_folder_itself(tmp_path):
    d = _split_dir(tmp_path)
    assert docker_entry.locate_split(d) == (d, "private_test")


def test_locate_split_refuses_two_splits(tmp_path):
    d = _split_dir(tmp_path)
    other = d.parent / "public_test"
    other.mkdir()
    (other / "public_test_tools_catalog.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        docker_entry.locate_split(d.parent)
    assert "private_test" in str(e.value) and "public_test" in str(e.value)


def test_fetch_models_skips_checkpoints(monkeypatch):
    import huggingface_hub

    seen = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **kw: seen.append(kw))
    docker_entry.fetch_models({"agent_preset": "sft-final", "asr_preset": "none", "agent_kw": {}, "asr_kw": {},
                               "hf_model_owner": "team"})
    assert seen and "checkpoints/*" in seen[0]["ignore_patterns"]


def test_check_models_fails_when_a_model_is_not_cached(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def missing(repo_id, **kw):
        assert kw["local_files_only"] is True
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(SystemExit) as e:
        docker_entry.check_models({"agent_preset": "sft-final", "asr_preset": "none", "agent_kw": {}, "asr_kw": {},
                                   "hf_model_owner": "team"})
    assert e.value.code not in (0, None)


def test_check_models_fails_on_a_partial_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")  # cached, but no config.json or weights
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **kw: str(tmp_path))
    with pytest.raises(SystemExit, match="incomplete snapshot"):
        docker_entry.check_models({"agent_preset": "sft-final", "asr_preset": "none", "agent_kw": {}, "asr_kw": {},
                                   "hf_model_owner": "team"}, load=False)


def test_smoke_config_names_known_presets():
    from lifemate.presets import resolve

    cfg = docker_entry.load_config(docker_entry.REPO_ROOT / "configs" / "smoke.json")
    env = {"HF_MODEL_OWNER": cfg["hf_model_owner"]}
    assert resolve("agent", cfg["agent_preset"], env)[0] == "empty"
    assert resolve("asr", cfg["asr_preset"], env)[0] == "none"


def test_committed_shipped_config_pins_both_models_by_full_commit():
    import re

    cfg = docker_entry.load_config()
    for part in ("agent_kw", "asr_kw"):
        assert re.fullmatch(r"[0-9a-f]{40}", cfg[part].get("revision") or ""), part  # a later push cannot change the system
