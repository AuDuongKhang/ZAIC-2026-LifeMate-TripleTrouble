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
