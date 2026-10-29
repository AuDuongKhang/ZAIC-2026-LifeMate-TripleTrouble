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
