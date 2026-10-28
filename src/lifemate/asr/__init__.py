"""ASR backends, picked by name with --asr. Add a backend: subclass ASR and register it in BACKENDS."""
from __future__ import annotations

from .base import ASR, GoldTranscriptASR, NullASR
from .cached import CachedASR
from .whisper import WhisperASR

BACKENDS: dict[str, type[ASR]] = {
    "cached": CachedASR,
    "gold": GoldTranscriptASR,
    "none": NullASR,
    "whisper": WhisperASR,
}


def build_asr(name: str, /, **kwargs) -> ASR:  # positional-only: the cached backend takes a `name` kwarg
    if name not in BACKENDS:
        raise ValueError(f"unknown ASR {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
