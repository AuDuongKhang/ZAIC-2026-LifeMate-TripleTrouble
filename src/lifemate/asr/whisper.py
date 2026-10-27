from __future__ import annotations

from pathlib import Path

from .base import ASR

SAMPLE_RATE = 16000


class WhisperASR(ASR):
    """Whisper-family checkpoint via the HF ASR pipeline (default PhoWhisper, Vietnamese), greedy decoding."""

    name = "whisper"

    def __init__(self, model: str = "vinai/PhoWhisper-medium", device: str = "cuda"):
        import torch
        from transformers import pipeline

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device, dtype=dtype)
        self.generate_kwargs = {"num_beams": 1, "do_sample": False}
        if getattr(self.pipe.model.generation_config, "lang_to_id", None):  # fine-tuned checkpoints may lack it
            self.generate_kwargs.update(language="vi", task="transcribe")

    def transcribe(self, audio: Path, sample: dict) -> str:
        import numpy as np
        import soundfile as sf

        wav, sr = sf.read(str(audio), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:  # the data is 16 kHz; linear resampling is only a fallback
            n = int(round(len(wav) * SAMPLE_RATE / sr))
            wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype("float32")
        out = self.pipe({"raw": wav, "sampling_rate": SAMPLE_RATE}, generate_kwargs=self.generate_kwargs)
        return out["text"].strip()
