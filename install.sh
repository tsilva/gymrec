#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

if uv tool list | grep -q "^gymrec "; then
  echo "Existing gymrec tool detected; upgrading editable tool install."
  uv tool install . -e --upgrade --force "$@"
else
  echo "Installing gymrec as an editable uv tool."
  uv tool install . -e "$@"
fi

uv tool run --from gymrec python - <<'PY'
from importlib.metadata import version

print(f"stable-retro-turbo {version('stable-retro-turbo')} installed")
PY
