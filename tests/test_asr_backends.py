import sys

import numpy as np
import pytest
import soundfile as sf

from lifemate.asr import build_asr
from lifemate.asr.base import SAMPLE_RATE, load_audio


def test_load_audio_downmixes_and_resamples(tmp_path):
    stereo = np.stack([np.linspace(-1, 1, 8000), np.linspace(-1, 1, 8000)], axis=1).astype("float32")
    sf.write(tmp_path / "a.wav", stereo, 8000)
    wav = load_audio(tmp_path / "a.wav")
    assert wav.ndim == 1 and wav.dtype == np.float32 and abs(len(wav) - SAMPLE_RATE) <= 1


def test_qwen3_asr_without_package_names_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "qwen_asr", None)  # import qwen_asr -> ImportError
    monkeypatch.setitem(sys.modules, "nagisa", None)
    with pytest.raises(ImportError, match="--no-deps qwen-asr"):
        build_asr("qwen3-asr")


def test_whisper_passes_the_revision(monkeypatch):
    import types

    from lifemate.asr.whisper import WhisperASR

    seen = {}

    def fake_pipeline(task, **kw):
        seen.update(kw)
        return types.SimpleNamespace(model=types.SimpleNamespace(generation_config=types.SimpleNamespace()))

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16="fp16", float32="fp32"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))
    WhisperASR(model="team/pho-large-ft-b", revision="abc123")
    assert seen["model"] == "team/pho-large-ft-b" and seen["revision"] == "abc123"
