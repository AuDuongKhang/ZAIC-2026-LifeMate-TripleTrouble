"""What an agent sees for one sample: transcribed request + history, context and the tool specs it may call."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    request: str            # ASR transcript of the final user turn (the one that is scored)
    history: list[dict]     # earlier turns; user turns get "text" (ASR), assistant turns keep text/tool_calls/tool_results
    context: dict           # current_time, timezone, location, user_profile, installed_apps
    tools: list[dict]       # catalog specs of the installed tools + sys.ask_missing_info
    label: dict | None = field(default=None, repr=False)  # train only; for oracle/debug agents, never at test time
    started_at: float | None = None  # time.perf_counter() at sample start, so agents can count ASR time against 30 s
