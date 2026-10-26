import zipfile

from lifemate.fetch_data import extract, has_data


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
