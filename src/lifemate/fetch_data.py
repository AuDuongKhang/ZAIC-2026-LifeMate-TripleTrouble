"""Download and unpack the competition data into data/ (train/, public_test/, README.md, VERSION).

    python -m lifemate.fetch_data --hf-repo <user>/<dataset>    # team copy (see lifemate.hf_upload); or LIFEMATE_HF_REPO
    python -m lifemate.fetch_data --url "<train+public_test.zip link>"   # organisers' server; or LIFEMATE_DATA_URL
    python -m lifemate.fetch_data --zip ~/Downloads/train+public_test.zip

The Hugging Face token comes from HF_TOKEN or `hf auth login`. Does nothing when the data is already there (use
--force to redo). Links and tokens are never stored in the repo: the data is licensed to registered teams only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .data import default_data_root

SPLITS = ("train", "public_test", "private_test")
HF_ZIP = "train+public_test.zip"  # file name inside the team's Hugging Face dataset


def has_data(dest: Path) -> bool:
    return any((dest / s / f"{s}.jsonl").is_file() for s in SPLITS)


def download(url: str, out: Path) -> None:
    with urllib.request.urlopen(url) as resp, open(out, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done, step = 0, 50 << 20
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done % step < len(chunk) or done == total:
                print(f"  {done >> 20} / {total >> 20} MB", flush=True)


def download_hf(repo_id: str, filename: str, out_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"downloading {repo_id}/{filename} from Hugging Face", flush=True)
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=out_dir))


def extract(zip_path: Path, dest: Path) -> None:
    """Unzip without macOS junk (__MACOSX/, .DS_Store) and without the zip's single top folder (train+public_test/)."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist()
                   if not m.filename.startswith("__MACOSX/") and Path(m.filename).name != ".DS_Store"]
        tops = {m.filename.split("/", 1)[0] for m in members}
        strip = len(tops) == 1 and not any(t in SPLITS for t in tops)
        for m in members:
            rel = m.filename.split("/", 1)[1] if strip and "/" in m.filename else m.filename
            if not rel or m.is_dir():
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="an already downloaded zip (checked first)")
    ap.add_argument("--hf-repo", default=os.environ.get("LIFEMATE_HF_REPO"), help="private HF dataset (checked second)")
    ap.add_argument("--hf-file", default=HF_ZIP)
    ap.add_argument("--url", default=os.environ.get("LIFEMATE_DATA_URL"), help="zip link (checked last)")
    ap.add_argument("--dest", type=Path, default=default_data_root())
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if has_data(args.dest) and not args.force:
        print(f"data already in {args.dest}")
        return args.dest
    if not (args.zip or args.hf_repo or args.url):
        sys.exit("no data yet: pass --hf-repo <user/dataset>, --url <zip link> or --zip <file>")
    args.dest.mkdir(parents=True, exist_ok=True)
    if args.zip:
        extract(args.zip, args.dest)
    else:
        with tempfile.TemporaryDirectory(dir=args.dest) as tmp:
            if args.hf_repo:
                zip_path = download_hf(args.hf_repo, args.hf_file, Path(tmp))
            else:
                zip_path = Path(tmp) / "data.zip"
                print(f"downloading -> {zip_path}", flush=True)
                download(args.url, zip_path)
            extract(zip_path, args.dest)
    if not has_data(args.dest):
        sys.exit(f"unzipped into {args.dest} but found no <split>/<split>.jsonl; check the zip layout")
    print(f"data ready in {args.dest}")
    return args.dest


if __name__ == "__main__":
    main()
