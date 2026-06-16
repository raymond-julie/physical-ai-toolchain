#!/usr/bin/env bash
# Native run helper for blob_sync.
#
#   ./run.sh            # watch mode (default)
#   ./run.sh --once     # one-shot
#   ./run.sh --check    # validate container access
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
if [[ ! -d "${VENV}" ]]; then
  echo "Creating virtualenv in ${VENV} ..."
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip >/dev/null
  "${VENV}/bin/pip" install -r requirements.txt
fi

exec "${VENV}/bin/python" -m blob_sync "$@"
