#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# smoke-acsa-mirror.sh
#
# Layer 1 isolation test for the LeRobot ACSA blob-sync path. Bypasses the
# recorder, the simulated camera, and the UR5e robots — proves only that:
#
#   1. The pvc-acsa-lerobot PVC is Bound.
#   2. A pod can write to /cloud-sync (the EdgeSubvolume mount).
#   3. ACSA asynchronously mirrors the write to the lerobot-recordings
#      blob container under the expected path.
#
# Why it exists
# -------------
# When the end-to-end pipeline fails (recorder writes locally but blobs never
# appear), the failure can be in: PVC binding, EdgeSubvolume reconciliation,
# managed-identity role assignment, network egress, or storage account naming.
# This test isolates ACSA + storage from everything downstream, so a failure
# here narrows the blame instantly.
#
# Idempotent: each run writes a timestamped marker file (one per invocation)
# so multiple runs leave a trail rather than racing on the same key.
#
# Required env:
#   STORAGE_ACCOUNT    Storage account name (printed by provision-lerobot-storage.sh).
#
# Optional env (defaults shown):
#   NAMESPACE          e2epyhsai
#   PVC_NAME           pvc-acsa-lerobot
#   CONTAINER          lerobot-recordings
#   REPO_ID            local/ur5_mirror
#   POD_IMAGE          mcr.microsoft.com/azurelinux/base/core:3.0
#   POLL_INTERVAL_SEC  20
#   POLL_TIMEOUT_SEC   600   (10 minutes — ACSA mirrors are typically <5 min)
#
# Exit codes:
#   0  Marker found in blob storage within POLL_TIMEOUT_SEC.
#   1  Marker not found within timeout (mirror failed or too slow).
#   2  Prerequisite missing (kubectl, az, PVC not bound, etc.).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NAMESPACE="${NAMESPACE:-e2epyhsai}"
PVC_NAME="${PVC_NAME:-pvc-acsa-lerobot}"
CONTAINER="${CONTAINER:-lerobot-recordings}"
REPO_ID="${REPO_ID:-local/ur5_mirror}"
POD_IMAGE="${POD_IMAGE:-mcr.microsoft.com/azurelinux/base/core:3.0}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-20}"
POLL_TIMEOUT_SEC="${POLL_TIMEOUT_SEC:-600}"
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:?STORAGE_ACCOUNT must be set (see provision-lerobot-storage.sh output)}"

# Per-invocation marker so concurrent / repeated runs do not race.
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RAND="$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n' || echo "xxxx")"
MARKER_BASENAME="smoke-${TS}-${RAND}.txt"
HOST_PATH_REL="${REPO_ID}/meta/${MARKER_BASENAME}"
HOST_PATH_ABS="/cloud-sync/${HOST_PATH_REL}"
BLOB_PATH="${REPO_ID}/meta/${MARKER_BASENAME}"
POD_NAME="acsa-smoke-${TS,,}-${RAND}"

echo "==> Namespace        : $NAMESPACE"
echo "==> PVC              : $PVC_NAME"
echo "==> Storage account  : $STORAGE_ACCOUNT"
echo "==> Container        : $CONTAINER"
echo "==> Marker (in-pod)  : $HOST_PATH_ABS"
echo "==> Marker (in-blob) : $BLOB_PATH"
echo "==> Poll timeout     : ${POLL_TIMEOUT_SEC}s (every ${POLL_INTERVAL_SEC}s)"
echo

# Prereq checks ──────────────────────────────────────────────────────────────
for cmd in kubectl az; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[error] required command not found: $cmd" >&2
        exit 2
    fi
done

# Confirm PVC is Bound before we even try to schedule the pod.
pvc_phase="$(kubectl get pvc -n "$NAMESPACE" "$PVC_NAME" \
    -o jsonpath='{.status.phase}' 2>/dev/null || echo "Missing")"
if [[ "$pvc_phase" != "Bound" ]]; then
    echo "[error] PVC $PVC_NAME in namespace $NAMESPACE has phase '$pvc_phase' (expected 'Bound')." >&2
    echo "        Run 'kubectl describe pvc -n $NAMESPACE $PVC_NAME' to investigate." >&2
    exit 2
fi
echo "[ok] PVC $PVC_NAME is Bound"

# Write the marker via a one-shot pod ────────────────────────────────────────
echo "[run] Writing marker via pod $POD_NAME (image $POD_IMAGE)"
overrides=$(cat <<JSON
{
  "spec": {
    "restartPolicy": "Never",
    "automountServiceAccountToken": false,
    "volumes": [
      {"name": "v", "persistentVolumeClaim": {"claimName": "$PVC_NAME"}}
    ],
    "containers": [
      {
        "name": "writer",
        "image": "$POD_IMAGE",
        "command": ["sh", "-c"],
        "args": [
          "set -e; mkdir -p \$(dirname $HOST_PATH_ABS); printf 'smoke %s\\n' '$TS' > $HOST_PATH_ABS; ls -la $HOST_PATH_ABS"
        ],
        "volumeMounts": [
          {"name": "v", "mountPath": "/cloud-sync"}
        ]
      }
    ]
  }
}
JSON
)

kubectl run "$POD_NAME" \
    --namespace "$NAMESPACE" \
    --image "$POD_IMAGE" \
    --restart=Never \
    --overrides="$overrides" \
    --command -- sh -c "true" >/dev/null

# Wait for completion (Succeeded or Failed). 60s should be more than enough
# for a one-shot write.
for _ in $(seq 1 60); do
    phase="$(kubectl get pod -n "$NAMESPACE" "$POD_NAME" \
        -o jsonpath='{.status.phase}' 2>/dev/null || echo "")"
    case "$phase" in
        Succeeded)
            echo "[ok] Writer pod completed"
            break
            ;;
        Failed)
            echo "[error] Writer pod failed:" >&2
            kubectl logs -n "$NAMESPACE" "$POD_NAME" >&2 || true
            kubectl delete pod -n "$NAMESPACE" "$POD_NAME" --wait=false >/dev/null 2>&1 || true
            exit 1
            ;;
    esac
    sleep 1
done

# Show what the pod actually wrote (helps debug path issues even if mirror works).
kubectl logs -n "$NAMESPACE" "$POD_NAME" 2>/dev/null || true
kubectl delete pod -n "$NAMESPACE" "$POD_NAME" --wait=false >/dev/null 2>&1 || true

# Poll blob storage for the marker ───────────────────────────────────────────
echo
echo "[wait] Polling blob storage for $BLOB_PATH"
deadline=$((SECONDS + POLL_TIMEOUT_SEC))
found=false
while (( SECONDS < deadline )); do
    if az storage blob exists \
            --account-name "$STORAGE_ACCOUNT" \
            --container-name "$CONTAINER" \
            --name "$BLOB_PATH" \
            --auth-mode login \
            --query exists -o tsv 2>/dev/null | grep -q '^true$'; then
        found=true
        break
    fi
    elapsed=$((SECONDS))
    echo "    [${elapsed}s] not yet — sleeping ${POLL_INTERVAL_SEC}s"
    sleep "$POLL_INTERVAL_SEC"
done

if ! $found; then
    echo
    echo "[fail] Marker did NOT appear in blob storage within ${POLL_TIMEOUT_SEC}s." >&2
    echo "       Investigate ACSA reconciliation:" >&2
    echo "         kubectl get edgesubvolume -n $NAMESPACE lerobot -o yaml" >&2
    echo "         kubectl logs -n azure-arc-containerstorage -l app=edge-storage-operator --tail=200" >&2
    echo "       Verify MI role assignment:" >&2
    echo "         az role assignment list --scope \$(az storage account show -n $STORAGE_ACCOUNT --query id -o tsv) -o table" >&2
    exit 1
fi

echo
echo "[pass] Marker mirrored to blob storage:"
az storage blob show \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER" \
    --name "$BLOB_PATH" \
    --auth-mode login \
    --query "{name:name, size:properties.contentLength, lastModified:properties.lastModified}" \
    -o table
echo
echo "ACSA blob-sync end-to-end OK."
