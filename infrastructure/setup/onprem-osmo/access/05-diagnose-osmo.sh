#!/usr/bin/env bash
#
# 05-diagnose-osmo.sh
#
# Pulls cluster-side diagnostics for the Osmo deployment so we can see
# why API calls return 503. Run this any time the UI shows
# "Unable to load workflows" or similar errors.
#
# What it does:
#   1. Lists all pods in the `osmo` namespace with status + restarts.
#   2. Describes any pod that isn't Running/Ready.
#   3. Tails the last log lines from key services (router, listener,
#      logger, service, ui, postgres).
#   4. Surfaces recent warning/error events in the namespace.
#
# All output is grouped with section headers so you can scroll/grep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_cmd az
ensure_logged_in
ensure_ssh_key

REMOTE_CMD='set +e; \
KCTL="sudo -n /usr/bin/kubectl --kubeconfig=/etc/kubernetes/admin.conf"; \
NS=osmo; \
echo "=== 1. Pods in $NS ==="; \
$KCTL get pods -n $NS -o wide; \
echo; \
echo "=== 2. Unhealthy pod descriptions ==="; \
BAD=$($KCTL get pods -n $NS --no-headers 2>/dev/null | awk "\$3 != \"Running\" && \$3 != \"Completed\" {print \$1}"); \
NOTREADY=$($KCTL get pods -n $NS --no-headers 2>/dev/null | awk "\$3 == \"Running\" {split(\$2,a,\"/\"); if (a[1] != a[2]) print \$1}"); \
ALL_BAD=$(printf "%s\n%s\n" "$BAD" "$NOTREADY" | sort -u | sed "/^$/d"); \
if [ -z "$ALL_BAD" ]; then echo "(all pods Running and Ready)"; \
else for p in $ALL_BAD; do \
  echo "--- describe $p ---"; \
  $KCTL describe pod -n $NS $p | sed -n "/Events:/,$ p" | head -40; \
  echo; \
done; fi; \
echo; \
echo "=== 3. Recent logs from key services (last 30 lines each) ==="; \
for label in app=osmo-router app=osmo-osmo-backend-listener app=osmo-logger app=osmo-service app=osmo-ui app=postgres; do \
  echo "--- $label ---"; \
  $KCTL logs -n $NS -l $label --tail=30 --all-containers 2>&1 | head -60; \
  echo; \
done; \
echo "=== 4. Recent warning/error events in $NS (last 25) ==="; \
$KCTL get events -n $NS --sort-by=.lastTimestamp 2>/dev/null | grep -Ei "warn|error|fail|backoff|unhealthy" | tail -25 || echo "(none)"; \
echo; \
echo "=== 5. Node pressure / resource summary ==="; \
$KCTL top nodes 2>/dev/null || echo "(metrics-server not available)"; \
$KCTL describe nodes 2>/dev/null | grep -A2 -E "Conditions:|MemoryPressure|DiskPressure|PIDPressure" | head -40'

az ssh arc \
  --subscription "${SUBSCRIPTION}" \
  --resource-group "${ARC_RESOURCE_GROUP}" \
  --name "${ARC_NAME}" \
  --local-user "${LOCAL_USER}" \
  --private-key-file "${SSH_KEY_PATH}" \
  -- "${REMOTE_CMD}"
