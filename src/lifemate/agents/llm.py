"""Chat LLM backends for agents. torch/transformers are imported in __init__ so tests run without them."""
from __future__ import annotations

from typing import Protocol

CONTEXT_LIMIT = 16384  # competition cap on the LLM context (prompt + generation)


class ChatLLM(Protocol):
    context_limit: int
    max_new_tokens: int

    def generate(self, messages: list[dict], tools: list[dict], max_time: float | None = None) -> str: ...

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int: ...


class HFChatLLM:
    """HF transformers model in its native chat template, greedy decoding. Returns the raw completion text,
    tool-call markup included (parsed by agents.parsing)."""

    def __init__(self, model: str = "Qwen/Qwen3-4B-Instruct-2507", max_new_tokens: int = 384, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.context_limit = CONTEXT_LIMIT
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype), device_map="cuda")
        self.model.eval()
        # Strings to cut from the completion: end-of-turn markers, not the <tool_call> markup.
        self.stop_strings = {t for t in (self.tokenizer.eos_token, self.tokenizer.pad_token, "<|im_end|>", "<|endoftext|>") if t}

    def _encode(self, messages: list[dict], tools: list[dict]):
        return self.tokenizer.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                                  return_tensors="pt", return_dict=True)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return int(self._encode(messages, tools)["input_ids"].shape[1])

    def generate(self, messages: list[dict], tools: list[dict], max_time: float | None = None) -> str:
        """max_time (seconds) stops decoding early; a cut-off tool call then fails to parse and is not sent."""
        inputs = self._encode(messages, tools).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                                      temperature=None, top_p=None, top_k=None, max_time=max_time)
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        for s in self.stop_strings:
            text = text.replace(s, "")
        return text.strip()
