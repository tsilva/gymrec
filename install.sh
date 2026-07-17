#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

echo "Installing or upgrading the editable gymrec tool."
uv tool install --project . . -e --upgrade --force "$@"

tool_path="$(uv tool dir --bin)/gymrec"
"$tool_path" --help >/dev/null
echo "gymrec installed successfully: $tool_path"
