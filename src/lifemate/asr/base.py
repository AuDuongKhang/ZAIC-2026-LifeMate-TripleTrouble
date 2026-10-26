from __future__ import annotations

from pathlib import Path


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
