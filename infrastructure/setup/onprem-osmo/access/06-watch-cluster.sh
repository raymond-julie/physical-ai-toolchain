#!/usr/bin/env bash
#
# 06-watch-cluster.sh
#
# Polls the on-prem Kubernetes cluster on `asus-nuc-4` until all nodes
# report Ready and the `osmo` namespace is healthy.
#
# Each tick prints:
#   * Per-node ping (LAN reachability, even before kubelet rejoins).
#   * `kubectl get nodes` summary.
#   * Pod health summary in the osmo namespace (ready/total + bad).
#
# Exits 0 once every node is Ready AND every osmo pod (excluding
# Completed jobs) is Running/Ready. Exits 1 on TIMEOUT_SECS.
#
# Configuration (env, all optional):
#   INTERVAL_SECS  poll interval                       (default 30)
#   TIMEOUT_SECS   give up after this long             (default 1800 = 30 min)
#   NODE_IPS       space-separated worker IPs to ping  (default discovered)

set -uo pipefail   # NB: no -e; we tolerate transient failures during reboot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_cmd az
ensure_logged_in
ensure_ssh_key

INTERVAL_SECS="${INTERVAL_SECS:-30}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
TICK_TIMEOUT_SECS="${TICK_TIMEOUT_SECS:-25}"
NODE_IPS="${NODE_IPS:-192.168.1.100 192.168.1.101 192.168.1.102 192.168.1.103}"

REMOTE_CMD='set +e; \
KCTL="sudo -n /usr/bin/kubectl --kubeconfig=/etc/kubernetes/admin.conf"; \
echo "--- ping ---"; \
for ip in '"${NODE_IPS}"'; do \
  if ping -c 1 -W 1 $ip >/dev/null 2>&1; then echo "  $ip  UP"; else echo "  $ip  --"; fi; \
done; \
echo "--- nodes ---"; \
$KCTL get nodes -o wide --no-headers 2>/dev/null | awk "{printf \"  %-45s %-12s %s\n\", \$1, \$2, \$6}"; \
echo "--- osmo pods ---"; \
TOTAL=$($KCTL get pods -n osmo --no-headers 2>/dev/null | grep -v Completed | wc -l); \
READY=$($KCTL get pods -n osmo --no-headers 2>/dev/null | awk "\$3==\"Running\" {split(\$2,a,\"/\"); if(a[1]==a[2]) print}" | wc -l); \
BAD=$($KCTL get pods -n osmo --no-headers 2>/dev/null | awk "\$3 != \"Running\" && \$3 != \"Completed\" {print \"  \" \$1 \"  \" \$3}"); \
echo "  ready: $READY / $TOTAL"; \
if [ -n "$BAD" ]; then echo "  not-running:"; echo "$BAD"; fi; \
NODES_NOTREADY=$($KCTL get nodes --no-headers 2>/dev/null | awk "\$2 != \"Ready\" {print \$1}" | wc -l); \
PODS_NOTREADY=$(echo "$BAD" | sed "/^$/d" | wc -l); \
if [ "$NODES_NOTREADY" -eq 0 ] && [ "$PODS_NOTREADY" -eq 0 ]; then echo "STATUS: HEALTHY"; else echo "STATUS: DEGRADED"; fi'

START=$(date +%s)
ATTEMPT=0
while :; do
  ATTEMPT=$((ATTEMPT+1))
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  if (( ELAPSED > TIMEOUT_SECS )); then
    echo
    echo "Timed out after ${TIMEOUT_SECS}s without reaching healthy state." >&2
    exit 1
  fi

  printf "\n=== tick #%d  elapsed=%ds  $(date '+%H:%M:%S') ===\n" "$ATTEMPT" "$ELAPSED"

  # Wrap the az ssh arc call so a rebooting master can't hang the loop.
  # `timeout` kills the process after TICK_TIMEOUT_SECS; SSH ConnectTimeout
  # short-circuits the TCP handshake when the host isn't responding yet.
  OUT=$(timeout --kill-after=5 "${TICK_TIMEOUT_SECS}" \
    az ssh arc \
    --subscription "${SUBSCRIPTION}" \
    --resource-group "${ARC_RESOURCE_GROUP}" \
    --name "${ARC_NAME}" \
    --local-user "${LOCAL_USER}" \
    --private-key-file "${SSH_KEY_PATH}" \
    -- -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
       "${REMOTE_CMD}" 2>&1)
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "  (master ${ARC_NAME} unreachable, rc=$RC -- waiting for it to come back)"
  else
    echo "$OUT"
  fi

  if echo "$OUT" | grep -q "STATUS: HEALTHY"; then
    echo
    echo "Cluster is healthy. Total wait: ${ELAPSED}s."
    exit 0
  fi

  sleep "${INTERVAL_SECS}"
done
