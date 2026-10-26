"""ASR backends, picked by name with --asr. Add a backend: subclass ASR and register it in BACKENDS."""
from __future__ import annotations

from .base import ASR, GoldTranscriptASR, NullASR

BACKENDS: dict[str, type[ASR]] = {
    "gold": GoldTranscriptASR,
    "none": NullASR,
}


def build_asr(name: str, **kwargs) -> ASR:
    if name not in BACKENDS:
        raise ValueError(f"unknown ASR {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
