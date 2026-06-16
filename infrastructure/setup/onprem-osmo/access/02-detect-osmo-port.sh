#!/usr/bin/env bash
#
# 02-detect-osmo-port.sh
#
# Connects to the NUC over Arc SSH and prints listening TCP ports together
# with the process holding them. Use the output to identify which port the
# Nvidia Osmo UI is bound to, then pass it to 03-tunnel-osmo.sh via the
# OSMO_PORT environment variable.
#
# Usage:
#   ./02-detect-osmo-port.sh
#   ./02-detect-osmo-port.sh | grep -i osmo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_cmd az
ensure_logged_in
ensure_ssh_key

REMOTE_CMD='set +e; \
echo "=== Listening TCP ports ==="; \
sudo -n ss -tlnp 2>/dev/null || ss -tln; \
echo; \
echo "=== Docker containers (if any) ==="; \
(command -v docker >/dev/null 2>&1 && (sudo -n docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}" 2>/dev/null || docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}" 2>/dev/null)) || echo "docker not available or not permitted"; \
echo; \
echo "=== Kubernetes services (all namespaces) ==="; \
KUBECTL=$(command -v kubectl || echo /usr/local/bin/kubectl); \
KUBECONFIG_TRY="${KUBECONFIG:-} /etc/kubernetes/admin.conf $HOME/.kube/config /var/lib/rancher/k3s/server/cred/admin.kubeconfig /etc/rancher/k3s/k3s.yaml"; \
KCTL=""; \
for kc in $KUBECONFIG_TRY; do \
  if [ -n "$kc" ] && (sudo -n test -r "$kc" 2>/dev/null || test -r "$kc"); then \
    KCTL="sudo -n $KUBECTL --kubeconfig=$kc"; \
    if ! sudo -n test -r "$kc" 2>/dev/null; then KCTL="$KUBECTL --kubeconfig=$kc"; fi; \
    break; \
  fi; \
done; \
if [ -z "$KCTL" ]; then echo "no readable kubeconfig found"; else \
  echo "(using: $KCTL)"; \
  $KCTL get svc -A 2>&1; \
  echo; \
  echo "=== Services with osmo/ui in name ==="; \
  $KCTL get svc -A 2>/dev/null | grep -iE "osmo|ui|web|portal|frontend|dashboard" || echo "none matched"; \
  echo; \
  echo "=== Ingresses ==="; \
  $KCTL get ingress -A 2>&1 || true; \
  echo; \
  echo "=== Pods (osmo/nvidia related) ==="; \
  $KCTL get pods -A 2>/dev/null | grep -iE "osmo|nvidia" || echo "no matching pods"; \
fi; \
echo; \
echo "=== Processes mentioning osmo ==="; \
ps -eo pid,user,cmd | grep -iE "osmo|nvidia" | grep -v grep || echo "no matching processes"'

az ssh arc \
  --subscription "${SUBSCRIPTION}" \
  --resource-group "${ARC_RESOURCE_GROUP}" \
  --name "${ARC_NAME}" \
  --local-user "${LOCAL_USER}" \
  --private-key-file "${SSH_KEY_PATH}" \
  -- "${REMOTE_CMD}"
