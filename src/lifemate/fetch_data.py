"""Download and unpack the competition data into data/ (train/, public_test/, README.md, VERSION).

    python -m lifemate.fetch_data --url "<train+public_test.zip link>"   # or set LIFEMATE_DATA_URL
    python -m lifemate.fetch_data --zip ~/Downloads/train+public_test.zip

Does nothing when the data is already there (use --force to redo). The link is not stored in the repo: the data is
licensed to registered teams only.
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
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--url", default=os.environ.get("LIFEMATE_DATA_URL"))
    src.add_argument("--zip", type=Path, help="an already downloaded zip")
    ap.add_argument("--dest", type=Path, default=default_data_root())
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if has_data(args.dest) and not args.force:
        print(f"data already in {args.dest}")
        return args.dest
    if not args.zip and not args.url:
        sys.exit("no data yet: pass --url <zip link> (or LIFEMATE_DATA_URL) or --zip <file>")
    args.dest.mkdir(parents=True, exist_ok=True)
    if args.zip:
        extract(args.zip, args.dest)
    else:
        with tempfile.TemporaryDirectory(dir=args.dest) as tmp:
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
