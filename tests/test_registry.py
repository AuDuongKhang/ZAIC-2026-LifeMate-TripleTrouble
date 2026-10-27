import subprocess
import sys

from lifemate.data import REPO_ROOT


def test_backends_registered_without_importing_torch():
    # Fresh interpreter: other tests (or plugins) may already have imported torch in this one.
    code = ("import sys; from lifemate.asr import BACKENDS; from lifemate.agents import AGENTS; "
            "assert 'whisper' in BACKENDS and 'react' in AGENTS, (sorted(BACKENDS), sorted(AGENTS)); "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT, env={"PYTHONPATH": str(REPO_ROOT / "src"),
                   "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")})
