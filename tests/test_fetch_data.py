import zipfile

from lifemate.fetch_data import extract, has_data
from lifemate.hf_upload import pack


def test_pack_then_extract_round_trips(tmp_path):
    src = tmp_path / "src"
    (src / "train" / "audio" / "t1").mkdir(parents=True)
    (src / "train" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (src / "train" / "audio" / "t1" / "t1.wav").write_bytes(b"RIFF")
    (src / "train" / ".DS_Store").write_bytes(b"junk")
    (src / "README.md").write_text("readme", encoding="utf-8")
    (src / "notes.txt").write_text("not part of the data", encoding="utf-8")
    dest = tmp_path / "dest"
    extract(pack(src, tmp_path / "data.zip"), dest)
    assert has_data(dest)
    assert (dest / "train" / "audio" / "t1" / "t1.wav").read_bytes() == b"RIFF"
    assert (dest / "README.md").is_file()
    assert not (dest / "train" / ".DS_Store").exists() and not (dest / "notes.txt").exists()


def test_extract_strips_top_folder_and_macos_junk(tmp_path):
    zp = tmp_path / "data.zip"
    with zipfile.ZipFile(zp, "w") as z:  # layout of the organisers' train+public_test.zip
        z.writestr("train+public_test/README.md", "readme")
        z.writestr("train+public_test/train/train.jsonl", "{}\n")
        z.writestr("train+public_test/train/.DS_Store", "junk")
        z.writestr("__MACOSX/train+public_test/._train", "junk")
    dest = tmp_path / "data"
    extract(zp, dest)
    assert has_data(dest)
    assert (dest / "README.md").is_file()
    assert not (dest / "__MACOSX").exists() and not (dest / "train" / ".DS_Store").exists()


def _split_zip(path, split):
    with zipfile.ZipFile(path, "w") as z:  # layout of the organisers' private_test.zip
        z.writestr("README.md", "split readme")
        z.writestr(f"{split}/{split}.jsonl", "{}\n")
        z.writestr(f"__MACOSX/{split}/._{split}.jsonl", "junk")
    return path


def _hf_files(tmp_path, files):
    """A fake download_hf serving `files` ({name: zip path}); other names raise as the hub does for a missing file."""
    from huggingface_hub.errors import EntryNotFoundError

    asked = []

    def download(repo_id, filename, out_dir):
        asked.append(filename)
        if filename not in files:
            raise EntryNotFoundError(f"{filename} not in {repo_id}")
        return files[filename]
    return download, asked


def test_fetch_adds_the_private_test_zip_to_data_already_there(tmp_path, monkeypatch):
    from lifemate import fetch_data

    dest = tmp_path / "data"
    (dest / "train").mkdir(parents=True)
    (dest / "train" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (dest / "README.md").write_text("data readme", encoding="utf-8")
    download, asked = _hf_files(tmp_path, {"private_test.zip": _split_zip(tmp_path / "p.zip", "private_test")})
    monkeypatch.setattr(fetch_data, "download_hf", download)
    fetch_data.main(["--hf-repo", "team/data", "--dest", str(dest)])
    assert (dest / "private_test" / "private_test.jsonl").is_file()
    assert asked == ["private_test.zip"]  # train+public_test.zip is not fetched again
    assert not (dest / "__MACOSX").exists()
    assert (dest / "README.md").read_text(encoding="utf-8") == "data readme"  # only the split folder is unpacked


def test_fetch_skips_a_split_zip_the_dataset_lacks_or_the_data_has(tmp_path, monkeypatch):
    from lifemate import fetch_data

    dest = tmp_path / "data"
    for split in ("train", "private_test"):
        (dest / split).mkdir(parents=True)
        (dest / split / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
    download, asked = _hf_files(tmp_path, {})
    monkeypatch.setattr(fetch_data, "download_hf", download)
    fetch_data.main(["--hf-repo", "team/data", "--dest", str(dest)])
    assert asked == []
    (dest / "private_test" / "private_test.jsonl").unlink()
    fetch_data.main(["--hf-repo", "team/data", "--dest", str(dest)])  # not uploaded yet: no error
    assert asked == ["private_test.zip"]
