"""Waveform augmentation for ASR fine-tuning (train clips only): speed perturbation, white noise and background speech.

Speed perturbation resamples the clip by a factor (0.9 = slower and lower, 1.1 = faster and higher), as in Kaldi's 3-way
speed perturbation. White noise is Gaussian at a random SNR. Background speech mixes another train clip, cropped or
tiled to the same length, at a random SNR, so no external audio is needed. Every choice comes from the `rng` passed in,
so a fixed seed gives the same augmented clips.
"""
from __future__ import annotations

from dataclasses import dataclass


def change_speed(wav, factor: float):
    """Resample `wav` so it plays `factor` times faster (length / factor samples), by linear interpolation."""
    import numpy as np

    if factor == 1.0 or len(wav) < 2:
        return wav
    n = max(2, int(round(len(wav) / factor)))
    src = np.linspace(0, len(wav) - 1, n)
    return np.interp(src, np.arange(len(wav)), wav).astype(wav.dtype)


def _power(x) -> float:
    import numpy as np

    return float(np.mean(np.square(x, dtype=np.float64)))


def add_at_snr(wav, noise, snr_db: float):
    """wav + noise scaled so that power(wav) / power(noise) = snr_db; noise is tiled or cropped to len(wav)."""
    import numpy as np

    if len(noise) == 0:
        return wav
    if len(noise) < len(wav):
        noise = np.tile(noise, int(np.ceil(len(wav) / len(noise))))
    noise = noise[: len(wav)]
    p_sig, p_noise = _power(wav), _power(noise)
    if p_sig == 0 or p_noise == 0:
        return wav
    scale = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    out = wav + scale * noise
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:  # keep it in [-1, 1] like a real recording
        out = out / peak
    return out.astype(wav.dtype)


@dataclass
class Augment:
    speeds: tuple[float, ...] = ()           # e.g. (0.9, 1.0, 1.1); empty = off
    noise_prob: float = 0.0                  # share of clips that get white noise
    babble_prob: float = 0.0                 # share of clips that get another clip underneath
    snr_db: tuple[float, float] = (10.0, 30.0)

    @property
    def active(self) -> bool:
        return bool(self.speeds) or self.noise_prob > 0 or self.babble_prob > 0

    def __call__(self, wav, rng, other=None):
        """Augment one clip. `rng` is a random.Random; `other()` returns another clip's waveform (background speech)."""
        import numpy as np

        if self.speeds:
            wav = change_speed(wav, rng.choice(self.speeds))
        if self.babble_prob > 0 and other is not None and rng.random() < self.babble_prob:
            wav = add_at_snr(wav, other(), rng.uniform(*self.snr_db))
        if self.noise_prob > 0 and rng.random() < self.noise_prob:
            noise = np.random.default_rng(rng.getrandbits(32)).standard_normal(len(wav)).astype(wav.dtype)
            wav = add_at_snr(wav, noise, rng.uniform(*self.snr_db))
        return wav


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())
