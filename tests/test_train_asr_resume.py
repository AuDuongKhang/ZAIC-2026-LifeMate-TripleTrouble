"""--resume of lifemate.train_asr picks the newest complete checkpoint, local or on the HF repo (no torch needed)."""
import json
import shutil
from pathlib import Path

import pytest

huggingface_hub = pytest.importorskip("huggingface_hub")
from lifemate.train_asr import resume_checkpoint  # noqa: E402


def _ckpt(folder: Path, step: int | None) -> Path:
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_text("", encoding="utf-8")
    if step is not None:  # without trainer_state.json the save was cut short
        (folder / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    return folder


@pytest.fixture
def hub(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()

    def fake_file(repo_id, filename, **kw):
        if not (root / filename).exists():
            raise FileNotFoundError(filename)
        return str(root / filename)

    def fake_snapshot(repo_id, allow_patterns, local_dir, **kw):
        shutil.copytree(root / "last-checkpoint", Path(local_dir) / "last-checkpoint")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_file)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    return root


def test_new_session_takes_the_hub_checkpoint(tmp_path, hub):
    _ckpt(hub / "last-checkpoint", 300)
    out = tmp_path / "run"
    _ckpt(out / "checkpoint-400", None)  # incomplete: ignored
    assert resume_checkpoint(out, "me/x") == out / "last-checkpoint"
    assert (out / "last-checkpoint" / "trainer_state.json").exists()


def test_newer_local_checkpoint_wins(tmp_path, hub):
    _ckpt(hub / "last-checkpoint", 300)
    out = tmp_path / "run"
    _ckpt(out / "checkpoint-300", 300)
    _ckpt(out / "checkpoint-600", 600)
    assert resume_checkpoint(out, "me/x") == out / "checkpoint-600"


def test_no_checkpoint_anywhere(tmp_path, hub):
    assert resume_checkpoint(tmp_path / "run", "me/x") is None
    assert resume_checkpoint(tmp_path / "run", None) is None
