"""Record, steps and pair files shared through the private HF dataset, with huggingface_hub replaced by fakes."""
import pytest

from lifemate.sft import hf_files


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake dataset: {path in repo: bytes}; downloads and uploads go through it."""
    files = {}
    remote = tmp_path / "remote"
    remote.mkdir()

    def download(repo_id, filename, repo_type, local_dir):
        from huggingface_hub.errors import EntryNotFoundError

        assert repo_type == "dataset" and repo_id == "o/data"
        if filename not in files:
            raise EntryNotFoundError(f"{filename} not found")
        path = remote / filename.replace("/", "_")
        path.write_bytes(files[filename])
        return str(path)

    def upload(path_or_fileobj, path_in_repo, repo_id, repo_type):
        assert repo_type == "dataset" and repo_id == "o/data"
        files[path_in_repo] = open(path_or_fileobj, "rb").read()

    monkeypatch.setattr(hf_files, "_download", download)
    monkeypatch.setattr(hf_files, "_upload_file", upload)
    return files


def test_push_then_pull_into_a_fresh_folder(tmp_path, hub):
    src, dest = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    (src / "pref-pilot-gold.jsonl").write_bytes(b'{"id": "train_0001"}\n')
    hf_files.main(["push", "--repo", "o/data", "--folder", "records", "--dir", str(src), "pref-pilot-gold.jsonl"])
    assert hub == {"records/pref-pilot-gold.jsonl": b'{"id": "train_0001"}\n'}
    hf_files.main(["pull", "--repo", "o/data", "--folder", "records", "--dir", str(dest), "pref-pilot-gold.jsonl"])
    assert (dest / "pref-pilot-gold.jsonl").read_bytes() == b'{"id": "train_0001"}\n'


def test_pull_keeps_a_local_file(tmp_path, hub):
    hub["records/x.jsonl"] = b"remote\n"
    (tmp_path / "x.jsonl").write_text("local\n", encoding="utf-8")
    hf_files.main(["pull", "--repo", "o/data", "--folder", "records", "--dir", str(tmp_path), "x.jsonl"])
    assert (tmp_path / "x.jsonl").read_text(encoding="utf-8") == "local\n"


def test_pull_of_a_file_the_dataset_lacks_stops_with_the_step_to_run(tmp_path, hub):
    with pytest.raises(SystemExit, match="records"):
        hf_files.main(["pull", "--repo", "o/data", "--folder", "records", "--dir", str(tmp_path), "missing.jsonl"])


def test_push_of_a_missing_local_file_stops(tmp_path, hub):
    with pytest.raises(SystemExit, match="nope.jsonl"):
        hf_files.main(["push", "--repo", "o/data", "--folder", "pairs", "--dir", str(tmp_path), "nope.jsonl"])
    assert hub == {}
