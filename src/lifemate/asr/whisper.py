from __future__ import annotations

from pathlib import Path

from .base import ASR, SAMPLE_RATE, load_audio
from .context import context_prompt


class WhisperASR(ASR):
    """Whisper-family checkpoint via the HF ASR pipeline (default PhoWhisper, Vietnamese); greedy decoding unless
    num_beams > 1."""

    name = "whisper"

    def __init__(self, model: str = "vinai/PhoWhisper-medium", device: str = "cuda", dtype: str = "float16",
                 revision: str | None = None, num_beams: int = 1, context_prompt: bool = False,
                 context_tokens: int = 150):
        """revision: HF repo commit, branch or tag of the model (None = latest). num_beams > 1 turns on beam search
        (deterministic, no sampling): slower per clip, often a little lower WER. context_prompt primes the decoder with
        the sample's profile and history (asr.context), keeping the last context_tokens tokens of it."""
        if num_beams < 1:
            raise ValueError(f"num_beams must be at least 1, got {num_beams}")
        import torch
        from transformers import pipeline

        torch_dtype = getattr(torch, dtype) if device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device, dtype=torch_dtype,
                             revision=revision)
        self.generate_kwargs = {"num_beams": int(num_beams), "do_sample": False}
        if getattr(self.pipe.model.generation_config, "lang_to_id", None):  # fine-tuned checkpoints may lack it
            self.generate_kwargs.update(language="vi", task="transcribe")
        self.use_context, self.context_tokens = context_prompt, int(context_tokens)

    def transcribe(self, audio: Path, sample: dict) -> str:
        wav = load_audio(audio)
        out = self.pipe({"raw": wav, "sampling_rate": SAMPLE_RATE}, generate_kwargs=self._generate_kwargs(sample))
        return out["text"].strip()

    def _generate_kwargs(self, sample: dict) -> dict:
        prompt = context_prompt(sample) if self.use_context else ""
        if not prompt:
            return self.generate_kwargs
        ids = self.pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
        if len(ids) > self.context_tokens:  # ids[0] is <|startofprev|>; the text nearest the audio is the latest
            ids = _join_ids(ids[:1], ids[1 - self.context_tokens:])
        return {**self.generate_kwargs, "prompt_ids": ids.to(self.pipe.device)}


def _join_ids(head, tail):
    import torch

    return torch.cat([head, tail])
