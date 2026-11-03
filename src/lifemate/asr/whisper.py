from __future__ import annotations

from pathlib import Path

from .base import ASR, SAMPLE_RATE, load_audio


class WhisperASR(ASR):
    """Whisper-family checkpoint via the HF ASR pipeline (default PhoWhisper, Vietnamese), greedy decoding."""

    name = "whisper"

    def __init__(self, model: str = "vinai/PhoWhisper-medium", device: str = "cuda", dtype: str = "float16",
                 revision: str | None = None):
        """revision: HF repo commit, branch or tag of the model (None = latest)."""
        import torch
        from transformers import pipeline

        torch_dtype = getattr(torch, dtype) if device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device, dtype=torch_dtype,
                             revision=revision)
        self.generate_kwargs = {"num_beams": 1, "do_sample": False}
        if getattr(self.pipe.model.generation_config, "lang_to_id", None):  # fine-tuned checkpoints may lack it
            self.generate_kwargs.update(language="vi", task="transcribe")

    def transcribe(self, audio: Path, sample: dict) -> str:
        wav = load_audio(audio)
        out = self.pipe({"raw": wav, "sampling_rate": SAMPLE_RATE}, generate_kwargs=self.generate_kwargs)
        return out["text"].strip()
