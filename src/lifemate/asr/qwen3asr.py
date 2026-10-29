from __future__ import annotations

import sys
import types
from pathlib import Path

from .base import ASR, SAMPLE_RATE, load_audio

INSTALL = "uv pip install --python <python> --no-deps qwen-asr  (plus librosa if it is missing)"


class Qwen3ASR(ASR):
    """Qwen3-ASR via the qwen-asr package (transformers backend), language forced to Vietnamese, greedy decoding.

    qwen-asr pins transformers and torch-related packages, so it is installed with --no-deps (see INSTALL) to keep the
    system torch. Count the real checkpoint size: 0.6B is 0.94B parameters, 1.7B is 2.35B (docs/asr_audit.md).
    """

    name = "qwen3-asr"

    def __init__(self, model: str = "Qwen/Qwen3-ASR-0.6B", device: str = "cuda", dtype: str = "bfloat16",
                 language: str = "Vietnamese", max_new_tokens: int = 256):
        try:
            import nagisa  # noqa: F401  (Japanese tokenizer, only used by the forced aligner)
        except ImportError:
            sys.modules["nagisa"] = types.ModuleType("nagisa")
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise ImportError(f"the qwen3-asr backend needs the qwen-asr package: {INSTALL}") from e
        import torch

        torch_dtype = getattr(torch, dtype) if device.startswith("cuda") else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(model, dtype=torch_dtype, device_map=device,
                                                   max_inference_batch_size=1, max_new_tokens=int(max_new_tokens))
        gen = self.model.model.generation_config  # the checkpoint already says do_sample=false; make it explicit
        gen.do_sample, gen.num_beams = False, 1
        self.language = language

    def transcribe(self, audio: Path, sample: dict) -> str:
        out = self.model.transcribe(audio=(load_audio(audio), SAMPLE_RATE), language=self.language)
        return out[0].text.strip()
