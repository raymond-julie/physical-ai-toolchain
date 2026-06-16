#!/usr/bin/env bash
#
# 03-tunnel-osmo.sh
#
# Opens an SSH local port-forward to the Nvidia Osmo UI on the NUC via
# `az ssh arc`.
#
# Osmo on `asus-nuc-4` runs on a local Kubernetes cluster behind an nginx
# ingress with host `quick-start.osmo`, exposed as the `quick-start`
# NodePort service (HTTP 30080, HTTPS 30443). This script forwards the
# HTTPS NodePort to a local port. While running, browse to:
#
#   https://${OSMO_INGRESS_HOST}:${LOCAL_BIND_PORT}
#
# Two prerequisites for that URL to resolve in your browser:
#   1. The ingress requires the `Host: quick-start.osmo` header. Map the
#      hostname to 127.0.0.1 by running ./04-add-hosts-entry.sh once
#      (or add the entry manually to your hosts file).
#   2. The ingress uses a self-signed cert; your browser will warn the
#      first time. Click through (or import the cert).
#
# Configuration (all optional, override via env):
#   OSMO_PORT          remote NodePort on the NUC      (default 30443)
#   LOCAL_BIND_PORT    local port to bind              (default 8443)
#
# Examples:
#   ./03-tunnel-osmo.sh
#   OSMO_PORT=30080 LOCAL_BIND_PORT=8080 ./03-tunnel-osmo.sh   # HTTP
#
# Stop the tunnel with Ctrl+C.
#
# Implementation notes / gotchas baked in:
#   * `az ssh arc` accepts pass-through OpenSSH flags after `--`. We pass
#     `-N` (no remote command), `-T` (no tty), and `-L` for the forward.
#   * `ServerAliveInterval`/`ExitOnForwardFailure` make the tunnel fail
#     loudly instead of silently if the remote port isn't actually open.
#   * If the local port is already in use, OpenSSH exits with code 255;
#     we detect that up front to give a clearer error message.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_cmd az ssh
ensure_logged_in
ensure_ssh_key

# Pre-flight: warn if the local bind port is already taken.
if command -v ss >/dev/null 2>&1; then
  if ss -tln "( sport = :${LOCAL_BIND_PORT} )" 2>/dev/null | grep -q LISTEN; then
    echo "Local port ${LOCAL_BIND_PORT} is already in use." >&2
    echo "Set LOCAL_BIND_PORT to a free port, e.g.:" >&2
    echo "  LOCAL_BIND_PORT=13000 $0" >&2
    exit 1
  fi
fi

if [[ "${OSMO_PORT}" == "30443" ]]; then SCHEME="https"; else SCHEME="http"; fi

cat <<EOF
Opening tunnel:
  remote  ${LOCAL_USER}@${ARC_NAME}:${OSMO_PORT}  (k8s NodePort -> ingress)
  local   ${SCHEME}://${OSMO_INGRESS_HOST}:${LOCAL_BIND_PORT}

If the URL fails to resolve, run ./04-add-hosts-entry.sh once (admin)
to map ${OSMO_INGRESS_HOST} -> 127.0.0.1.

Press Ctrl+C to disconnect.
EOF

# `--` separates az ssh arc args from OpenSSH args.
# Auto-reconnect loop: the Arc relay (sshproxy) occasionally drops with
# `wsarecv: A connection attempt failed...`. We restart the tunnel until
# the user hits Ctrl+C.
trap 'echo; echo "Tunnel stopped."; exit 0' INT TERM

while :; do
  az ssh arc \
    --subscription "${SUBSCRIPTION}" \
    --resource-group "${ARC_RESOURCE_GROUP}" \
    --name "${ARC_NAME}" \
    --local-user "${LOCAL_USER}" \
    --private-key-file "${SSH_KEY_PATH}" \
    -- \
      -N -T \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -L "${LOCAL_BIND_PORT}:localhost:${OSMO_PORT}" \
    || true
  echo "[tunnel dropped at $(date '+%H:%M:%S'); reconnecting in 3s...]"
  sleep 3
done
