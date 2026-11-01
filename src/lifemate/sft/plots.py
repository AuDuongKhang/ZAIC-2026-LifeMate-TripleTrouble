"""Training plots from train_log.jsonl: loss.png (train and eval), lr.png, grad_norm.png. The molab training notebook
redraws them while a run trains, and train.py saves them with every checkpoint."""
from __future__ import annotations

import json
from pathlib import Path


def read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plot_log(log_path: Path, out_dir: Path) -> list[Path]:
    """The PNGs written; none when matplotlib is missing or the log is empty."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: no plots", flush=True)
        return []
    rows = read_log(log_path)
    if not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, keys in (("loss", ("loss", "eval_loss")), ("lr", ("lr",)), ("grad_norm", ("grad_norm",))):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for key in keys:
            pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), marker="o" if key.startswith("eval") else None, label=key)
        ax.set_xlabel("optimizer step")
        ax.set_title(name)
        if ax.lines:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        written.append(out_dir / f"{name}.png")
    return written
