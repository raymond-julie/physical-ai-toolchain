#!/usr/bin/env bash
#
# common.sh
#
# Shared configuration for the on-prem OSMO remote-access scripts. Sourced by
# the other scripts; not meant to be executed directly.
#
# Every value below is overridable from the environment. The defaults match
# the Houston ("hou") on-prem training cell (an Azure Arc-enabled ASUS NUC
# running a bare-metal kubeadm cluster). Point the scripts at a different
# cell by exporting the matching variables before invoking them, e.g.:
#
#   ARC_NAME=my-edge-box LOCAL_USER=ops RESOURCE_GROUP=rg-my-cell \
#     ./01-create-vnet.sh

# --- Azure target ------------------------------------------------------------
export SUBSCRIPTION="${SUBSCRIPTION:-57b15bf0-e8dd-458a-9156-0694edd7ad4e}"
export RESOURCE_GROUP="${RESOURCE_GROUP:-rg-hou-physical-ai-training-cell-eai}"
export LOCATION="${LOCATION:-southcentralus}"

# --- Arc machine (the on-prem NUC running Osmo) ------------------------------
export ARC_RESOURCE_GROUP="${ARC_RESOURCE_GROUP:-rg-hou-facility}"
export ARC_NAME="${ARC_NAME:-asus-nuc-4}"
export LOCAL_USER="${LOCAL_USER:-edge}"
export SSH_KEY_PATH="${SSH_KEY_PATH:-${HOME}/.ssh/arc_${ARC_NAME//-/_}_${LOCAL_USER}}"

# --- VNet layout -------------------------------------------------------------
export VNET_NAME="${VNET_NAME:-vnet-hou-physical-ai-training}"
export VNET_ADDRESS_SPACE="${VNET_ADDRESS_SPACE:-10.42.0.0/16}"
export SUBNET_DEFAULT_NAME="${SUBNET_DEFAULT_NAME:-snet-default}"
export SUBNET_DEFAULT_PREFIX="${SUBNET_DEFAULT_PREFIX:-10.42.1.0/24}"

# --- Osmo UI -----------------------------------------------------------------
# On `asus-nuc-4`, Osmo runs on Kubernetes (namespace `osmo`). The UI is
# served behind an nginx ingress with host `quick-start.osmo`, exposed via
# the `quick-start` NodePort service:
#   HTTP  30080  ->  ingress :80
#   HTTPS 30443  ->  ingress :443
# Browsers must send the `Host: quick-start.osmo` header for the ingress to
# route to the UI, so we map that hostname locally via /etc/hosts (or
# C:\Windows\System32\drivers\etc\hosts on Windows).
export OSMO_INGRESS_HOST="${OSMO_INGRESS_HOST:-quick-start.osmo}"
export OSMO_PORT="${OSMO_PORT:-30443}"          # remote NodePort on the NUC
export LOCAL_BIND_PORT="${LOCAL_BIND_PORT:-8443}" # local browser port

# --- Helpers -----------------------------------------------------------------
require_cmd() {
  for c in "$@"; do
    if ! command -v "$c" >/dev/null 2>&1; then
      echo "Required command not found: $c" >&2
      exit 1
    fi
  done
}

ensure_logged_in() {
  if ! az account show --subscription "${SUBSCRIPTION}" >/dev/null 2>&1; then
    echo "Logging in to Azure subscription ${SUBSCRIPTION}..."
    az login --output none
  fi
  az account set --subscription "${SUBSCRIPTION}"
}

ensure_ssh_key() {
  if [[ ! -f "${SSH_KEY_PATH}" ]]; then
    echo "SSH key for the NUC not found at ${SSH_KEY_PATH}." >&2
    echo "Run ./setup-ssh-key.sh first." >&2
    exit 1
  fi
}
