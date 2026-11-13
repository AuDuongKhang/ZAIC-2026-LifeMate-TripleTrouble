"""Share record, steps and pair files through the team's private HF dataset, so a new molab session does not rebuild them.

    python -m lifemate.sft.hf_files push --repo <owner>/<dataset> --folder records --dir train_records pref-r1-gold.jsonl
    python -m lifemate.sft.hf_files pull --repo <owner>/<dataset> --folder records --dir train_records pref-r1-gold.jsonl

`push` uploads each file to <folder>/<name> in the dataset (token: HF_TOKEN, write) and stops on a missing local file.
`pull` downloads each file the folder does not have yet (a local copy always wins) and stops, naming the step that
writes it, when the dataset does not have it either. The files hold train data, so they only go to the private dataset.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

WRITTEN_BY = {"records": "records", "steps": "sample", "pairs": "pairs"}


def _download(repo_id: str, filename: str, repo_type: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir)


def _upload_file(path_or_fileobj: str, path_in_repo: str, repo_id: str, repo_type: str) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(path_or_fileobj=path_or_fileobj, path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type)


def pull(repo: str, folder: str, dest: Path, names: list[str]) -> None:
    from huggingface_hub.errors import EntryNotFoundError

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            if (dest / name).exists():
                print(f"{dest / name}: local copy kept", flush=True)
                continue
            try:
                got = _download(repo_id=repo, filename=f"{folder}/{name}", repo_type="dataset", local_dir=tmp)
            except EntryNotFoundError:
                raise SystemExit(f"{repo} has no {folder}/{name}: run the `{WRITTEN_BY.get(folder, folder)}` step first "
                                 "(with the same name and fields)") from None
            shutil.copyfile(got, dest / name)
            print(f"pulled {repo}/{folder}/{name}", flush=True)


def push(repo: str, folder: str, src: Path, names: list[str]) -> None:
    missing = [name for name in names if not (src / name).exists()]
    if missing:
        raise SystemExit(f"nothing to upload for {', '.join(missing)} in {src}")
    for name in names:
        _upload_file(path_or_fileobj=str(src / name), path_in_repo=f"{folder}/{name}", repo_id=repo, repo_type="dataset")
        print(f"pushed {src / name} -> {repo}/{folder}/{name}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("pull", "push"))
    ap.add_argument("--repo", required=True, help="private HF dataset (owner/name)")
    ap.add_argument("--folder", required=True, help="folder in the dataset: records, steps or pairs")
    ap.add_argument("--dir", type=Path, required=True, help="local folder")
    ap.add_argument("names", nargs="+")
    args = ap.parse_args(argv)
    (pull if args.action == "pull" else push)(args.repo, args.folder, args.dir, args.names)


if __name__ == "__main__":
    main()
