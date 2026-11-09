import random

import numpy as np
import pytest

from lifemate.asr_augment import Augment, add_at_snr, change_speed, parse_floats


def _tone(n=16000, freq=440.0):
    return (0.3 * np.sin(2 * np.pi * freq * np.arange(n) / 16000)).astype(np.float32)


def test_change_speed_scales_length_and_keeps_dtype():
    wav = _tone()
    assert len(change_speed(wav, 0.9)) == round(16000 / 0.9)
    assert len(change_speed(wav, 1.1)) == round(16000 / 1.1)
    assert change_speed(wav, 1.1).dtype == np.float32
    assert change_speed(wav, 1.0) is wav


def test_add_at_snr_hits_the_requested_ratio():
    wav = _tone()
    noise = np.random.default_rng(0).standard_normal(8000).astype(np.float32)  # shorter: gets tiled
    out = add_at_snr(wav, noise, 20.0)
    assert len(out) == len(wav)
    added = out - wav
    snr = 10 * np.log10(np.mean(wav.astype(np.float64) ** 2) / np.mean(added.astype(np.float64) ** 2))
    assert snr == pytest.approx(20.0, abs=0.01)


def test_add_at_snr_keeps_peak_within_one_and_skips_silence():
    loud = np.ones(100, dtype=np.float32) * 0.9
    assert np.max(np.abs(add_at_snr(loud, np.ones(100, dtype=np.float32), 0.0))) <= 1.0 + 1e-6
    silent = np.zeros(100, dtype=np.float32)
    assert add_at_snr(silent, np.ones(100, dtype=np.float32), 10.0) is silent


def test_augment_is_off_by_default_and_deterministic_for_a_seed():
    wav = _tone()
    assert not Augment().active
    assert Augment()(wav, random.Random(0)) is wav
    aug = Augment(speeds=(0.9, 1.0, 1.1), noise_prob=1.0, babble_prob=1.0)
    other = lambda: _tone(freq=300.0)  # noqa: E731
    a = aug(wav, random.Random("0:5:0"), other)
    b = aug(wav, random.Random("0:5:0"), other)
    c = aug(wav, random.Random("0:5:1"), other)
    assert np.array_equal(a, b)
    assert not (len(a) == len(c) and np.array_equal(a, c))


def test_babble_needs_another_clip():
    wav = _tone()
    out = Augment(babble_prob=1.0)(wav, random.Random(0), None)  # no other clip: unchanged
    assert out is wav


def test_parse_floats():
    assert parse_floats("0.9,1.0, 1.1") == (0.9, 1.0, 1.1)
    assert parse_floats("") == ()


class _Proc:  # the two processor calls ClipDataset makes
    def __init__(self):
        self.feature_extractor = lambda wav, sampling_rate: type("F", (), {"input_features": [np.asarray(wav)]})()
        self.tokenizer = lambda text: type("T", (), {"input_ids": [1, 2]})()


def _dataset(augment):
    from lifemate.train_asr import ClipDataset

    ds = ClipDataset([{"audio": "a", "text": "x"}, {"audio": "b", "text": "y"}], None, _Proc(), augment, seed=0)
    ds._wav = lambda i: _tone(freq=440.0 + 100 * i)
    return ds


def test_clip_dataset_without_augment_returns_the_clip():
    ds = _dataset(None)
    assert np.array_equal(ds[0]["input_features"], _tone())


def test_clip_dataset_augments_differently_on_each_read_and_repeats_for_a_fresh_run():
    aug = Augment(speeds=(0.9, 1.1), noise_prob=1.0, babble_prob=1.0)
    ds = _dataset(aug)
    first, second = ds[0]["input_features"], ds[0]["input_features"]
    assert not (len(first) == len(second) and np.array_equal(first, second))
    again = _dataset(aug)[0]["input_features"]
    assert np.array_equal(first, again)
