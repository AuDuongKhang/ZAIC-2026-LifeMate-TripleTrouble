"""Few-shot examples from our own training data: for each request, the most similar solved training requests whose
apps are all installed, shown as worked tool calls in the system prompt. The pool (configs/splits.json) is used for
experiments, so dev and holdout stay unseen; public_test and the private test use every train sample."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from ..asr.wer import normalize
from ..data import load_catalog, load_samples
from ..splits import load_splits
from .refs import scalar_strings, unseen_refs

K1, B = 1.5, 0.75


def _gold_apps(sample: dict) -> set[str]:
    """Apps of the gold calls, including the app of the tool a sys.ask_missing_info asks about."""
    apps = set()
    for call in sample["label"]["tool_calls"]:
        name = call["arguments"].get("tool_name", "") if call["name"] == "sys.ask_missing_info" else call["name"]
        apps.add(name.split(".", 1)[0])
    return apps - {"sys", ""}


def _sees_its_ids(sample: dict, catalog: dict[str, dict]) -> bool:
    """Every id the gold calls pass (x-ref arguments) appears in an earlier result of the example itself. The example
    is shown without its history, so an id from the history would teach the model to use ids from nowhere."""
    seen: set[str] = set()
    for call, result in zip(sample["label"]["tool_calls"], sample["label"]["tool_results"]):
        if call["name"] != "sys.ask_missing_info" and unseen_refs(catalog.get(call["name"]), call["arguments"], seen):
            return False
        scalar_strings(result, seen)
    return True


class ExampleBank:
    def __init__(self, samples: list[dict], pool_ids: set[str], catalog: dict[str, dict] | None = None):
        """With a tool catalog, samples whose ids come from their history rather than their own results are left out."""
        self.docs = sorted((s for s in samples if s["id"] in pool_ids and (catalog is None or _sees_its_ids(s, catalog))),
                           key=lambda s: s["id"])
        self.tokens = [Counter(normalize(s["audio_text"])) for s in self.docs]
        self.apps = [_gold_apps(s) for s in self.docs]
        self.lengths = [sum(t.values()) for t in self.tokens]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(w for t in self.tokens for w in t)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def __len__(self) -> int:
        return len(self.docs)

    @classmethod
    def from_data(cls, data_root: Path, split: str = "train") -> "ExampleBank":
        """The pool when evaluating on train (dev/holdout stay unseen); every train sample for any other split."""
        samples = load_samples(data_root, "train")
        pool = set(load_splits()["pool"]) if split == "train" else {s["id"] for s in samples}
        return cls(samples, pool, load_catalog(data_root, "train"))

    def _bm25(self, i: int, query: set[str]) -> float:
        tf, norm = self.tokens[i], 1 - B + B * self.lengths[i] / max(self.avg_len, 1e-9)
        return sum(self.idf[w] * tf[w] * (K1 + 1) / (tf[w] + K1 * norm) for w in query if w in tf)

    def select(self, request: str, installed_apps: set[str], k: int, exclude_id: str | None = None) -> list[dict]:
        """Up to k pool samples whose gold-call apps are all installed (out-of-scope ones always qualify), best
        BM25 match to the request first, ties by id. exclude_id is never returned."""
        query = set(normalize(request))
        ranked = sorted((-self._bm25(i, query), doc["id"], i) for i, doc in enumerate(self.docs)
                        if doc["id"] != exclude_id and self.apps[i] <= installed_apps)
        return [self.docs[i] for _, _, i in ranked[:k]]


def render_examples(examples: list[dict], max_result_chars: int = 300) -> str:
    if not examples:
        return ""
    lines = ["Examples of correct behaviour on other requests (the ids and values belong to those examples, not to "
             "this conversation):"]
    for ex in examples:
        label = ex["label"]
        lines += ["", f"User: {ex['audio_text']}"]
        if not label["tool_calls"]:
            lines.append("(no tool call)")
        for call, result in zip(label["tool_calls"], label["tool_results"]):
            lines.append("<tool_call>" + json.dumps({"name": call["name"], "arguments": call["arguments"]},
                                                     ensure_ascii=False) + "</tool_call>")
            text = json.dumps(result, ensure_ascii=False)
            lines.append("-> " + (text[:max_result_chars] + "..." if len(text) > max_result_chars else text))
    return "\n".join(lines)
