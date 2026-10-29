from __future__ import annotations

from pathlib import Path

SAMPLE_RATE = 16000


def load_audio(audio: Path):
    """Mono float32 waveform at 16 kHz (the data is 16 kHz; linear resampling is only a fallback)."""
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(str(audio), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(wav) * SAMPLE_RATE / sr))
        wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype("float32")
    return wav


class ASR:
    """Speech -> Vietnamese text. One instance is reused for every sample (load models in __init__)."""

    name = "base"

    def transcribe(self, audio: Path, sample: dict) -> str:
        raise NotImplementedError


class NullASR(ASR):
    """Empty transcripts: for agents that ignore the request (empty baseline, format checks) on any split."""

    name = "none"

    def transcribe(self, audio: Path, sample: dict) -> str:
        return ""


class GoldTranscriptASR(ASR):
    """Train only: returns the labelled audio_text. Gives the agent's upper bound with perfect ASR."""

    name = "gold"

    def transcribe(self, audio: Path, sample: dict) -> str:
        if "audio_text" not in sample:
            raise ValueError(f"{sample['id']}: no audio_text (gold ASR only works on train)")
        if Path(sample["audio"]).name == audio.name:
            return sample["audio_text"]
        for turn in sample["history"]:
            if turn.get("audio") and Path(turn["audio"]).name == audio.name:
                return turn["audio_text"]
        raise KeyError(f"{sample['id']}: no transcript for {audio.name}")
