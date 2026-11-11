#!/usr/bin/env bash
# Stream a built image to Google Drive as <folder>/zac2026_TripleTrouble.tar.gz (the name the guide asks for), with
# an info.txt next to it, share it (anyone with the link) and write md5/bytes/link/folder files to OUT_DIR.
#   OUT_DIR=/tmp/out bash scripts/drive_upload.sh zac2026-tripletrouble:<sha> gdrive:zaic2026/<unique folder>
# Every upload gets its own folder, so earlier links (one of them may be the submitted one) never change. The folder
# must not exist yet. The stream is retried (3 attempts); a partial file is deleted before each retry. The MD5 is the
# one Drive computed on the stored file, i.e. what the organisers download.
set -euo pipefail
image="$1"
dest="$2"
archive="${ARCHIVE:-zac2026_TripleTrouble.tar.gz}"
out="${OUT_DIR:?set OUT_DIR}"
mkdir -p "$out"

if [ -n "$(rclone lsf "$dest" 2>/dev/null)" ]; then
  echo "refusing to upload: $dest already has files" >&2
  exit 1
fi

gz=gzip
command -v pigz >/dev/null && gz=pigz
for attempt in 1 2 3; do
  echo "upload attempt $attempt -> $dest/$archive"
  if docker save "$image" | "$gz" | rclone rcat "$dest/$archive" --streaming-upload-cutoff 0 \
       --drive-chunk-size 256M --stats 60s --stats-one-line -v; then
    break
  fi
  rclone deletefile "$dest/$archive" 2>/dev/null || true
  if [ "$attempt" = 3 ]; then
    echo "upload failed 3 times" >&2
    exit 1
  fi
  sleep 30
done

md5="$(rclone md5sum "$dest/$archive" | cut -d' ' -f1)"
bytes="$(rclone size --json "$dest/$archive" | jq -r '.bytes')"
link="$(rclone link "$dest/$archive")"
case "$md5" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
  *) echo "Drive returned no MD5 for $dest/$archive" >&2; exit 1 ;;
esac

{
  echo "archive: $archive"
  echo "md5: $md5"
  echo "bytes: $bytes"
  echo "link: $link"
  echo "image: $image ($(docker image inspect --format '{{.Id}}' "$image"))"
  echo "commit: ${GITHUB_SHA:-$(git rev-parse HEAD)}"
  echo "run: ${GITHUB_SERVER_URL:-}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID:-local}"
  echo "uploaded: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$out/info.txt" | rclone rcat "$dest/info.txt"

echo "$md5  $archive" > "$out/$archive.md5"
echo "$bytes" > "$out/bytes.txt"
echo "$link" > "$out/link.txt"
echo "$dest" > "$out/folder.txt"
echo "uploaded $dest/$archive ($bytes bytes, md5 $md5)"
