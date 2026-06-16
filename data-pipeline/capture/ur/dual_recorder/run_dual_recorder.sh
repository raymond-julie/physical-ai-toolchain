#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# UrDualRecorder — launcher
#
# Dual-arm UR recorder. Reads its robot/camera topology from
# /etc/trainmybot/config_v3.yaml, reads both follower arms' joint + Robotiq
# 2F-85 gripper state over RTDE (read-only), captures the cameras, and records
# LeRobot episodes. No teleoperation/mirroring.
#
# Usage:
#   ./run_dual_recorder.sh                  # GUI + recording (DI0 trigger on)
#   ./run_dual_recorder.sh --no-web         # headless
#   ./run_dual_recorder.sh --no-record      # preview only
#   ./run_dual_recorder.sh --no-di0-trigger # GUI Record button only
#   ./run_dual_recorder.sh --help           # passthrough to the Python CLI
# ─────────────────────────────────────────────────────────────────────────
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="${TRAINMYBOT_CONFIG:-/etc/trainmybot/config_v3.yaml}"
APP_CONFIG="${APP_CONFIG:-$SCRIPT_DIR/config/app.yaml}"

# ── Pick the LeRobot dataset format (and matching venv) ──────────────────
# Priority: LEROBOT_FORMAT env > recording.lerobot_format in app.yaml > v2.1.
# v2.1 → .venv (lerobot 0.3.3), v3.0 → .venv-v3 (lerobot 0.4.4).
LEROBOT_FORMAT="${LEROBOT_FORMAT:-}"
if [[ -z "$LEROBOT_FORMAT" && -f "$APP_CONFIG" ]]; then
  # Minimal YAML read: an uncommented `lerobot_format: <value>` line.
  LEROBOT_FORMAT="$(sed -n 's/^[[:space:]]*lerobot_format:[[:space:]]*["'\'']\?\([^"'\''#[:space:]]*\).*/\1/p' "$APP_CONFIG" | head -n1)"
fi
LEROBOT_FORMAT="${LEROBOT_FORMAT:-v2.1}"
export LEROBOT_FORMAT

case "$LEROBOT_FORMAT" in
  v3.0|v3) VENV_DIR="$SCRIPT_DIR/.venv-v3" ;;
  *)       VENV_DIR="$SCRIPT_DIR/.venv" ;;
esac

# Activate the chosen venv if present, otherwise fall back to .venv.
if [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -d "$SCRIPT_DIR/.venv" ]]; then
  echo "WARNING: $VENV_DIR not found; falling back to .venv" >&2
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -f "$APP_CONFIG" ]]; then
  EXTRA_ARGS+=(--app-config "$APP_CONFIG")
fi

echo "Config:     $CONFIG_PATH"
[[ -f "$APP_CONFIG" ]] && echo "App config: $APP_CONFIG"
echo "Format:     $LEROBOT_FORMAT  (venv: $(basename "${VIRTUAL_ENV:-system}"))"

exec python3 -m ur_dual_recorder --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}" "$@"
