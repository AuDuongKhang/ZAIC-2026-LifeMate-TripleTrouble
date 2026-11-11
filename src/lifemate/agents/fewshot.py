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
NEAR_TOOLS = 3  # installed tools of apps the bank has no example of, nearest to the request, that an example may resemble


def gold_apps(label: dict) -> set[str]:
    """Apps of the gold calls of a label, including the app of the tool a sys.ask_missing_info asks about."""
    apps = set()
    for call in label["tool_calls"]:
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


def tool_tokens(spec: dict) -> set[str]:
    """The shape of a tool, free of its app: the words of its name after the app (end_session -> end, session), of its
    parameter names, and its kind (read / write)."""
    words = set(spec["name"].split(".", 1)[-1].lower().split("_"))
    for param in ((spec.get("parameters") or {}).get("properties") or {}):
        words |= set(param.lower().split("_"))
    if spec.get("kind"):
        words.add(spec["kind"])
    return {w for w in words if w}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def gold_tool_names(sample: dict) -> list[str]:
    """The tools of the gold calls (for an ask, the tool it asks about), in order, without repeats."""
    names = []
    for call in sample["label"]["tool_calls"]:
        name = call["arguments"].get("tool_name", "") if call["name"] == "sys.ask_missing_info" else call["name"]
        if name and name not in names:
            names.append(name)
    return names


def extract_unseen_history(samples: list[dict], catalog: dict[str, dict], known_apps: set[str]) -> list[dict]:
    """Worked examples mined from `samples`'s `history` turns for apps absent from `known_apps`.

    Uses only assistant turn fields (`text`, `tool_calls`, `tool_results`), which exist in every split (including
    `public_test` and `private_test`) without running ASR. When a write/ask turn has an earlier read-only turn on the
    same app in the same history, they are combined so the example shows the full read -> write/ask sequence and
    satisfies `_sees_its_ids`. Keeps every self-contained write/ask example plus read-only examples for tools not
    already covered by a write/ask example. Each doc keeps its parent sample `id`, so `select(..., exclude_id=task.id)`
    never shows a sample its own history."""
    raw: list[tuple[tuple[str, int], dict, bool]] = []
    for s in samples:
        h = s.get("history") or []
        asst = [h[i + 1] for i in range(len(h) - 1)
                if h[i].get("role") == "user" and h[i + 1].get("role") == "assistant" and h[i + 1].get("tool_calls")]
        used: set[tuple[str, int]] = set()
        for idx, a in enumerate(asst):
            calls, results = a["tool_calls"], a["tool_results"]
            doc = {"id": s["id"], "audio_text": a.get("text", ""),
                   "label": {"tool_calls": calls, "tool_results": results}}
            apps = gold_apps(doc["label"])
            if not apps or apps <= known_apps:
                continue
            is_wa = any(c["name"] == "sys.ask_missing_info" or (catalog.get(c["name"]) or {}).get("kind") == "write"
                        for c in calls)
            if is_wa:
                for p_idx in range(idx - 1, -1, -1):
                    pa = asst[p_idx]
                    pcalls, presults = pa["tool_calls"], pa["tool_results"]
                    pdoc = {"label": {"tool_calls": pcalls, "tool_results": presults}}
                    p_ro = all(c["name"] != "sys.ask_missing_info"
                               and (catalog.get(c["name"]) or {}).get("kind") == "read" for c in pcalls)
                    if p_ro and gold_apps(pdoc["label"]) == apps and (p_idx == idx - 1 or not _sees_its_ids(doc, catalog)):
                        comb = {"id": s["id"], "audio_text": (pa.get("text", "") + " " + a.get("text", "")).strip(),
                                "label": {"tool_calls": pcalls + calls, "tool_results": presults + results}}
                        if _sees_its_ids(comb, catalog):
                            doc = comb
                            used.add((s["id"], p_idx))
                            break
            if _sees_its_ids(doc, catalog):
                raw.append(((s["id"], idx), doc, is_wa))
        raw = [(k, d, is_wa) for k, d, is_wa in raw if k not in used]
    wa_tools = {c["name"] for _, d, is_wa in raw if is_wa for c in d["label"]["tool_calls"]}
    return [d for _, d, is_wa in raw if is_wa or any(c["name"] not in wa_tools for c in d["label"]["tool_calls"])]


class ExampleBank:
    def __init__(self, samples: list[dict], pool_ids: set[str], catalog: dict[str, dict] | None = None):
        """With a tool catalog, samples whose ids come from their history rather than their own results are left out."""
        self.docs = sorted((s for s in samples if s["id"] in pool_ids and (catalog is None or _sees_its_ids(s, catalog))),
                           key=lambda s: s["id"])
        self.tokens = [Counter(normalize(s["audio_text"])) for s in self.docs]
        self.apps = [gold_apps(s["label"]) for s in self.docs]
        self.bank_apps = set().union(*self.apps) if self.apps else set()
        self.catalog = catalog
        self.example_tools = [[tool_tokens({**catalog[n], "name": n}) for n in gold_tool_names(s) if n in catalog]
                              for s in self.docs] if catalog else []
        self.lengths = [sum(t.values()) for t in self.tokens]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(w for t in self.tokens for w in t)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def __len__(self) -> int:
        return len(self.docs)

    @classmethod
    def from_data(cls, data_root: Path, split: str = "train", history_examples: bool = False) -> "ExampleBank":
        """The pool when evaluating on train (dev/holdout stay unseen); every train sample for any other split.
        With `history_examples=True`, also mines self-contained history turns for apps absent from the training bank:
        from the pool's histories when evaluating on train (so dev/holdout stay unseen), from `split` otherwise
        (unseen apps on public_test or private_test)."""
        samples = load_samples(data_root, "train")
        train_cat = load_catalog(data_root, "train")
        pool = set(load_splits()["pool"]) if split == "train" else {s["id"] for s in samples}
        if not history_examples:
            return cls(samples, pool, train_cat)
        base_docs = [s for s in samples if s["id"] in pool and _sees_its_ids(s, train_cat)]
        known_apps = set().union(*(gold_apps(s["label"]) for s in base_docs)) if base_docs else set()
        # On train, mine only pool histories: dev/holdout histories repeat the scenarios of other dev/holdout
        # samples (same templates, same env), so mining them would leak those samples' answers into the gate.
        split_samples = [s for s in samples if s["id"] in pool] if split == "train" else load_samples(data_root, split)
        split_cat = train_cat if split == "train" else load_catalog(data_root, split)
        hdocs = extract_unseen_history(split_samples, split_cat, known_apps)
        all_docs = base_docs + hdocs
        return cls(all_docs, {d["id"] for d in all_docs}, {**train_cat, **split_cat})

    def _bm25(self, i: int, query: set[str]) -> float:
        tf, norm = self.tokens[i], 1 - B + B * self.lengths[i] / max(self.avg_len, 1e-9)
        return sum(self.idf[w] * tf[w] * (K1 + 1) / (tf[w] + K1 * norm) for w in query if w in tf)

    def _near_tools(self, query: set[str], tools: list[dict]) -> list[set[str]]:
        """tool_tokens of the NEAR_TOOLS installed tools of apps absent from the bank that share the most words with the
        request (the words of their description and name), ties by name. A tool sharing no word is not near."""
        scored = []
        for spec in tools:
            app = spec["name"].split(".", 1)[0]
            if app == "sys" or app in self.bank_apps:
                continue
            words = set(normalize(spec.get("description", "") + " " + spec["name"].split(".", 1)[-1].replace("_", " ")))
            if query & words:
                scored.append((-len(query & words), spec["name"], spec))
        return [tool_tokens(spec) for _, _, spec in sorted(scored, key=lambda t: t[:2])[:NEAR_TOOLS]]

    def select(self, request: str, installed_apps: set[str], k: int, exclude_id: str | None = None,
               offset: int = 0, tools: list[dict] | None = None, analog_weight: float = 0.0,
               exclude_apps: "frozenset[str] | set[str]" = frozenset()) -> list[dict]:
        """Up to k pool samples whose gold-call apps are all installed (out-of-scope ones always qualify), best
        BM25 match to the request first, ties by id. exclude_id is never returned. offset skips that many of the
        best matches (the prompt vote shows each variant the next k). Docs whose gold apps intersect exclude_apps are
        skipped (out-of-scope docs have no gold apps, so they still qualify).

        analog_weight > 0 adds analog_weight x the mean, over the example's gold tools, of the best Jaccard of
        tool_tokens against the NEAR_TOOLS installed tools of apps the bank has no example of (`tools` are the
        installed specs). It finds a tool of the same shape on another app, such as end_ride for end_session. With
        no such tool the score is plain BM25, so a sample whose apps are all in the bank keeps its examples."""
        query = set(normalize(request))
        near = []
        if analog_weight and tools:
            if self.catalog is None:
                raise ValueError("analog_weight needs the tool catalog (ExampleBank(..., catalog=...))")
            near = self._near_tools(query, tools)

        def score(i: int) -> float:
            bm25 = self._bm25(i, query)
            if not near or not self.example_tools[i]:
                return bm25
            return bm25 + analog_weight * sum(max(_jaccard(t, n) for n in near)
                                              for t in self.example_tools[i]) / len(self.example_tools[i])

        ranked = sorted((-score(i), doc["id"], i) for i, doc in enumerate(self.docs)
                        if doc["id"] != exclude_id and self.apps[i] <= installed_apps and not self.apps[i] & exclude_apps)
        return [self.docs[i] for _, _, i in ranked[offset:offset + k]]


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
