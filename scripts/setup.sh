#!/usr/bin/env bash
# Create .venv and install the project (macOS / Linux):  bash scripts/setup.sh
# Then: source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."

# macOS ships python3 3.9 with the Xcode tools; we need >= 3.10 (3.11 recommended for ML libs)
PY=""
for cand in python3.11 python3.12 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Need Python >= 3.10. On macOS: brew install python@3.11" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating .venv with $("$PY" --version)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-dev.txt -e .
echo "Installed into .venv ($(python --version))"

if python -c 'import sys; from lifemate.fetch_data import has_data; from lifemate.data import default_data_root; sys.exit(not has_data(default_data_root()))'; then
  echo "Data found in data/. Check: pytest"
else
  echo "No data yet. Run: python -m lifemate.fetch_data --url \"<zip link>\"  (or --zip <downloaded zip>)"
fi
echo "Activate with: source .venv/bin/activate"
