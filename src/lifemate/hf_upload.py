"""Put the competition data in the team's PRIVATE Hugging Face dataset, once, so molab sessions fetch it fast.

    hf auth login                                                    # or set HF_TOKEN (write access)
    python -m lifemate.hf_upload --repo <user>/zaic2026-lifemate-data --zip ~/Downloads/train+public_test.zip
    python -m lifemate.hf_upload --repo <user>/zaic2026-lifemate-data    # re-packs the unpacked data/ folder

The dataset is created private and the upload is refused if it is public: the data is licensed to registered
teams only. Read it back with `python -m lifemate.fetch_data --hf-repo <user>/zaic2026-lifemate-data`.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

from .data import default_data_root
from .fetch_data import HF_ZIP, SPLITS, has_data


def pack(data_dir: Path, out: Path) -> Path:
    """Zip the split folders + README.md/VERSION of an unpacked data/ (no macOS junk). Audio is stored, not deflated."""
    roots = [data_dir / s for s in SPLITS if (data_dir / s).is_dir()] + \
            [data_dir / f for f in ("README.md", "VERSION") if (data_dir / f).is_file()]
    with zipfile.ZipFile(out, "w") as z:
        for root in roots:
            files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
            for p in files:
                if p.name == ".DS_Store" or "__pycache__" in p.parts:
                    continue
                kind = zipfile.ZIP_STORED if p.suffix == ".wav" else zipfile.ZIP_DEFLATED
                z.write(p, p.relative_to(data_dir).as_posix(), compress_type=kind)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="<user or org>/<dataset name>")
    ap.add_argument("--zip", type=Path, help="the organisers' zip; default: pack --data-dir")
    ap.add_argument("--data-dir", type=Path, default=default_data_root())
    ap.add_argument("--path-in-repo", default=HF_ZIP)
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        sys.exit(f"{args.repo} is PUBLIC; make it private on huggingface.co before uploading competition data")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = args.zip
        if zip_path is None:
            if not has_data(args.data_dir):
                sys.exit(f"no data in {args.data_dir}; pass --zip or run lifemate.fetch_data first")
            print(f"packing {args.data_dir} ...", flush=True)
            zip_path = pack(args.data_dir, Path(tmp) / HF_ZIP)
        print(f"uploading {zip_path} ({zip_path.stat().st_size >> 20} MB) -> {args.repo}/{args.path_in_repo}", flush=True)
        api.upload_file(path_or_fileobj=str(zip_path), path_in_repo=args.path_in_repo, repo_id=args.repo,
                        repo_type="dataset", commit_message="Add competition data")
    print(f"done: python -m lifemate.fetch_data --hf-repo {args.repo}")


if __name__ == "__main__":
    main()
