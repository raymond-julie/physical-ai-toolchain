#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# provision-lerobot-storage.sh
#
# Provision the Azure Blob Storage account + container that the Arc Container
# Storage extension (ACSA) mirrors the LeRobot recordings PVC to. Grants the
# ACSA extension's managed identity `Storage Blob Data Owner` on the storage
# account scope so the EdgeSubvolume can write without keys.
#
# Idempotent: re-running is safe.
#
# Required env:
#   CLUSTER_NAME       Arc-connected cluster name (no default).
#
# Optional env (defaults shown):
#   RESOURCE_GROUP     rg-schaeffler-robotics
#   LOCATION           westus3
#   STORAGE_ACCOUNT    derived from RESOURCE_GROUP md5; override for stability
#   CONTAINER          lerobot-recordings
#   EXTENSION_NAME     azure-arc-containerstorage
#
# Prints `STORAGE_ACCOUNT_ENDPOINT=https://...` on success — substitute that
# value into `schaeffler_robotics_gitops/apps/lerobot_recorder_wl/edgesubvolume.yaml`
# (or wire it through Flux variable substitution) before applying the manifest.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-schaeffler-robotics}"
LOCATION="${LOCATION:-westus3}"
CONTAINER="${CONTAINER:-lerobot-recordings}"
EXTENSION_NAME="${EXTENSION_NAME:-azure-arc-containerstorage}"
CLUSTER_NAME="${CLUSTER_NAME:?CLUSTER_NAME must be set (Arc-connected cluster)}"

# Derive a deterministic-but-unique storage account name when not overridden.
# Storage account names: 3-24 chars, lowercase letters + digits only.
if [[ -z "${STORAGE_ACCOUNT:-}" ]]; then
    rg_hash=$(printf '%s' "$RESOURCE_GROUP" | md5sum | cut -c1-6)
    STORAGE_ACCOUNT="schaefflerlerobot${rg_hash}"
fi

echo "==> Resource group : $RESOURCE_GROUP"
echo "==> Location       : $LOCATION"
echo "==> Storage account: $STORAGE_ACCOUNT"
echo "==> Container      : $CONTAINER"
echo "==> Cluster        : $CLUSTER_NAME"
echo "==> Extension      : $EXTENSION_NAME"
echo

# 1) Storage account ──────────────────────────────────────────────────────────
if az storage account show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$STORAGE_ACCOUNT" >/dev/null 2>&1; then
    echo "[ok] Storage account $STORAGE_ACCOUNT already exists"
else
    echo "[create] Storage account $STORAGE_ACCOUNT"
    az storage account create \
        --name "$STORAGE_ACCOUNT" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        --kind StorageV2 \
        --enable-hierarchical-namespace false \
        --min-tls-version TLS1_2 \
        --allow-blob-public-access false \
        >/dev/null
fi

STORAGE_ID=$(az storage account show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$STORAGE_ACCOUNT" \
    --query id -o tsv)

# 2) Container ────────────────────────────────────────────────────────────────
if az storage container show \
        --account-name "$STORAGE_ACCOUNT" \
        --name "$CONTAINER" \
        --auth-mode login >/dev/null 2>&1; then
    echo "[ok] Container $CONTAINER already exists"
else
    echo "[create] Container $CONTAINER"
    az storage container create \
        --account-name "$STORAGE_ACCOUNT" \
        --name "$CONTAINER" \
        --auth-mode login \
        >/dev/null
fi

# 3) Role assignment for the ACSA extension MI ───────────────────────────────
PRINCIPAL_ID=$(az k8s-extension show \
    --cluster-name "$CLUSTER_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --cluster-type connectedClusters \
    --name "$EXTENSION_NAME" \
    --query identity.principalId -o tsv)

if [[ -z "$PRINCIPAL_ID" ]]; then
    echo "[error] Could not resolve managed identity principalId for extension $EXTENSION_NAME"
    echo "        Verify the Arc Container Storage extension is installed on $CLUSTER_NAME."
    exit 1
fi

if az role assignment list \
        --assignee "$PRINCIPAL_ID" \
        --scope "$STORAGE_ID" \
        --role "Storage Blob Data Owner" \
        --query "[].id" -o tsv | grep -q .; then
    echo "[ok] Role 'Storage Blob Data Owner' already granted to MI $PRINCIPAL_ID"
else
    echo "[grant] 'Storage Blob Data Owner' on $STORAGE_ACCOUNT to MI $PRINCIPAL_ID"
    az role assignment create \
        --assignee "$PRINCIPAL_ID" \
        --role "Storage Blob Data Owner" \
        --scope "$STORAGE_ID" \
        >/dev/null
fi

# 4) Emit the endpoint for the EdgeSubvolume manifest ─────────────────────────
ENDPOINT=$(az storage account show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$STORAGE_ACCOUNT" \
    --query primaryEndpoints.blob -o tsv)

echo
echo "================================================================"
echo "STORAGE_ACCOUNT_ENDPOINT=$ENDPOINT"
echo "================================================================"
echo "Substitute the value above into:"
echo "  schaeffler_robotics_gitops/apps/lerobot_recorder_wl/edgesubvolume.yaml"
echo "(or expose it via Flux variable substitution as STORAGE_ACCOUNT_ENDPOINT)."
