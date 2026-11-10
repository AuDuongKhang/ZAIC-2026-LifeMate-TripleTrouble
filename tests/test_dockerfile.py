"""Text checks of the image recipe against the organiser Docker guide."""
import re

from lifemate.data import REPO_ROOT

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split("."))


def test_base_image_has_cuda_at_most_13_2_and_torch_for_the_rtx_5090():
    m = re.search(r"^ARG BASE=pytorch/pytorch:(\d+\.\d+\.\d+)-cuda(\d+\.\d+)-", DOCKERFILE, re.M)
    assert m, "ARG BASE must name a pytorch/pytorch:<torch>-cuda<x.y>-... image"
    torch_v, cuda_v = _version(m.group(1)), _version(m.group(2))
    assert torch_v >= (2, 7, 0)  # first release with sm_120 (RTX 5090) kernels
    assert (12, 8) <= cuda_v <= (13, 2)  # organiser driver 595.58.03 runs CUDA <= 13.2
    assert "'sm_80' in a and 'sm_120' in a" in DOCKERFILE  # build fails without A30 and RTX 5090 kernels


def test_image_uses_the_code_layout():
    assert re.search(r"^WORKDIR /code$", DOCKERFILE, re.M)
    for name in ("docker/predict.sh", "docker/start_jupyter.sh", "docker/predict_notebook.ipynb",
                 "docker/training_code"):
        assert name in DOCKERFILE
    assert "jupyterlab" in DOCKERFILE and "nbconvert" in DOCKERFILE
    assert "HF_HUB_OFFLINE=1" in DOCKERFILE


def test_build_script_writes_the_md5():
    script = (REPO_ROOT / "scripts" / "docker_build.sh").read_text(encoding="utf-8")
    assert "zac2026_TripleTrouble.tar.gz" in script and "md5sum" in script


def test_workflow_never_reuses_a_drive_folder_and_queues_runs():
    wf = (REPO_ROOT / ".github" / "workflows" / "docker-image.yml").read_text(encoding="utf-8")
    # an upload run waits for the previous one instead of killing it; only PR checks cancel their older run
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in wf
    assert "${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT}" in wf  # one folder per run and attempt
    upload = (REPO_ROOT / "scripts" / "drive_upload.sh").read_text(encoding="utf-8")
    assert "refusing to upload" in upload and "rclone md5sum" in upload


def test_shipped_fingerprint_holds_the_shipped_presets():
    import json
    import subprocess
    import sys

    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "shipped_fingerprint.py"), "HEAD"], cwd=REPO_ROOT,
                       capture_output=True, text=True, encoding="utf-8", check=True)
    fp = json.loads(r.stdout)
    shipped = json.loads((REPO_ROOT / "configs" / "shipped.json").read_text(encoding="utf-8"))
    presets = json.loads((REPO_ROOT / "configs" / "presets.json").read_text(encoding="utf-8"))
    assert fp["presets"]["agent"] == presets["agent"][shipped["agent_preset"]]
    assert fp["presets"]["asr"] == presets["asr"][shipped["asr_preset"]]


def test_pull_requests_build_without_uploading():
    wf = (REPO_ROOT / ".github" / "workflows" / "docker-image.yml").read_text(encoding="utf-8")
    assert "pull_request:" in wf and "configs/shipped.json" in wf
    assert "UPLOAD: ${{ github.event_name == 'push' || (github.event_name == 'workflow_dispatch' && inputs.upload) }}" in wf


def test_dockerignore_keeps_the_data_package_out():
    ignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "data/*" in ignore and "!data/train/train.jsonl" in ignore and "context" in ignore
