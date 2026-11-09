"""Chat LLM backends for agents. torch/transformers are imported in __init__ so tests run without them."""
from __future__ import annotations

from typing import Protocol

CONTEXT_LIMIT = 16384  # competition cap on the LLM context (prompt + generation)


class ChatLLM(Protocol):
    context_limit: int
    max_new_tokens: int

    def generate(self, messages: list[dict], tools: list[dict], max_time: float | None = None) -> str: ...

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int: ...


def finish_text(text: str, stop_strings: set[str], open_think: bool) -> str:
    """The completion as the agent should read it: the stop strings removed; when it holds `</think>`, only what follows
    the last one (a call written inside the thinking is a draft and must never reach env.call); and nothing at all when
    a thinking block was never closed, because the token limit cut it off: one the prompt opened (open_think) or one the
    model opened itself (the completion starts with <think>)."""
    for stop in stop_strings:
        text = text.replace(stop, "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif open_think or text.lstrip().startswith("<think>"):  # a model that opens the block itself and is cut off too
        return ""
    return text.strip()


END_OF_TURN = ("<|im_end|>", "<|endoftext|>")


def end_of_turn_ids(tokenizer, generation_config=None) -> list[int]:
    """Token ids that end a completion: the tokenizer's and the generation config's EOS plus the chat end-of-turn markers
    that exist in the vocabulary. Qwen3.5 ships no generation_config.json and its config names only <|endoftext|> as EOS,
    so without <|im_end|> here generate() runs past the end of the turn and writes a made-up next turn (tool results
    included) until max_new_tokens."""
    ids: list[int] = []

    def add(x):
        for i in (x if isinstance(x, (list, tuple)) else [x]):
            if isinstance(i, int) and i >= 0 and i not in ids:
                ids.append(i)

    add(getattr(tokenizer, "eos_token_id", None))
    if generation_config is not None:
        add(getattr(generation_config, "eos_token_id", None))
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for token in END_OF_TURN:
        if token in vocab:
            add(vocab[token])
    return ids


def load_causal_lm(model: str, dtype, revision: str | None = None):
    """AutoModelForCausalLM.from_pretrained that refuses a checkpoint with missing weights. A multimodal checkpoint (for
    example Qwen3.5, whose text weights live under model.language_model.*) loads into a text model with other parameter
    names without raising, and every weight that finds no match stays randomly initialised."""
    from transformers import AutoModelForCausalLM

    loaded, info = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, revision=revision, output_loading_info=True)
    if info["missing_keys"]:
        missing = sorted(info["missing_keys"])
        raise RuntimeError(f"{model}: {len(missing)} weights are missing from the checkpoint (for example {missing[:3]}), "
                           "so they would be random; a multimodal checkpoint must go through lifemate.sft.text_only")
    return loaded


class HFChatLLM:
    """HF transformers model in its native chat template, greedy decoding. Returns the raw completion text,
    tool-call markup included (parsed by agents.parsing)."""

    def __init__(self, model: str = "Qwen/Qwen3-4B-Instruct-2507", max_new_tokens: int = 384, dtype: str = "bfloat16",
                 revision: str | None = None, num_beams: int = 1):
        """revision: HF repo commit, branch or tag of the model (None = latest). num_beams > 1 runs beam search
        (still deterministic)."""
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.context_limit = CONTEXT_LIMIT
        self.max_new_tokens, self.num_beams = max_new_tokens, num_beams
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        # .to() instead of device_map="cuda": device_map needs accelerate, which would pull its own torch
        self.model = load_causal_lm(model, getattr(torch, dtype), revision).to("cuda")
        self.model.eval()
        # Strings to cut from the completion: end-of-turn markers, not the <tool_call> markup.
        self.stop_strings = {t for t in (self.tokenizer.eos_token, self.tokenizer.pad_token, *END_OF_TURN) if t}
        self.eos_ids = end_of_turn_ids(self.tokenizer, getattr(self.model, "generation_config", None))
        pad = getattr(self.tokenizer, "pad_token_id", None)
        self.pad_id = pad if pad is not None else (self.eos_ids[0] if self.eos_ids else None)

    def _encode(self, messages: list[dict], tools: list[dict]):
        return self.tokenizer.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                                  return_tensors="pt", return_dict=True)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return int(self._encode(messages, tools)["input_ids"].shape[1])

    def generate(self, messages: list[dict], tools: list[dict], max_time: float | None = None) -> str:
        """max_time (seconds) stops decoding early; a cut-off tool call then fails to parse and is not sent."""
        inputs = self._encode(messages, tools).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=self.num_beams,
                                      temperature=None, top_p=None, top_k=None, max_time=max_time,
                                      eos_token_id=self.eos_ids or None, pad_token_id=self.pad_id)
        prompt_len = inputs["input_ids"].shape[1]
        tail = self.tokenizer.decode(inputs["input_ids"][0, -8:], skip_special_tokens=False)
        open_think = tail.endswith("<think>\n")  # the prompt opened a thinking block the output has to close
        text = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=False)
        return finish_text(text, self.stop_strings, open_think)
