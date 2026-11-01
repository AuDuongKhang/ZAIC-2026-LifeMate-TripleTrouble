from __future__ import annotations

from ..task import Task


class Agent:
    """Handles one sample by calling env.call(name, arguments). Every env.call is logged and scored (failed ones count
    against precision), so check arguments with env.validate (free, not logged) before calling. One instance is reused
    for every sample (load models in __init__); create no ToolEnv here, the runner passes the sample's only one."""

    name = "base"
    # Per-sample diagnostics for trace.jsonl, reset at the start of every run; None for agents that keep none.
    last_trace: dict | None = None

    def run(self, task: Task, env) -> None:
        raise NotImplementedError


class EmptyAgent(Agent):
    """Calls nothing: the out-of-scope answer for every sample. Floor score / format check."""

    name = "empty"

    def run(self, task: Task, env) -> None:
        return None


class OracleAgent(Agent):
    """Train only: replays the gold calls. Must score 100 on train, which checks the runner + scorer."""

    name = "oracle"

    def run(self, task: Task, env) -> None:
        if task.label is None:
            raise ValueError(f"{task.id}: oracle agent needs labels (train only)")
        for call in task.label["tool_calls"]:
            env.call(call["name"], call["arguments"])
