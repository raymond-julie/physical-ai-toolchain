#!/usr/bin/env bash
# Launch the camera streaming service.
#
# Usage:
#   ./run_streamer.sh                       # auto-discover, serve on :8000
#   ./run_streamer.sh --port 9000
#   ./run_streamer.sh --config /etc/trainmybot/config_v3.yaml
#   ./run_streamer.sh --list                # list cameras and exit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Use a local venv if present, otherwise the system/user Python.
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python3 -m camera_streamer "$@"
