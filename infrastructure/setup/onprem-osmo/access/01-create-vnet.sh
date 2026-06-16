#!/usr/bin/env bash
#
# 01-create-vnet.sh
#
# Idempotently creates the VNet and a default subnet in the training-cell
# resource group (default: rg-hou-physical-ai-training-cell-eai). Re-running
# is safe; existing resources are detected and left alone.
#
# Usage:
#   ./01-create-vnet.sh
#   ./01-create-vnet.sh --config-preview   # print resolved config and exit
#
# Why a VNet at all (since Arc SSH port-forwarding works without one)?
#   - Reserves an address space for future training-cell resources
#     (storage private endpoints, jumpboxes, AKS, etc.) without renumbering.
#   - Keeps the training-cell footprint discoverable in one resource group.
#
# All identity/network values are overridable from the environment — see
# common.sh for the full list (SUBSCRIPTION, RESOURCE_GROUP, LOCATION,
# VNET_NAME, VNET_ADDRESS_SPACE, SUBNET_DEFAULT_NAME, SUBNET_DEFAULT_PREFIX).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Idempotently create the training-cell VNet and default subnet.

OPTIONS:
    -h, --help          Show this help message
    --config-preview    Print the resolved configuration and exit
EOF
}

config_preview=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)        show_help; exit 0 ;;
    --config-preview) config_preview=true; shift ;;
    *)                echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "${config_preview}" == "true" ]]; then
  echo "=== Configuration Preview ==="
  echo "Subscription   : ${SUBSCRIPTION}"
  echo "Resource group : ${RESOURCE_GROUP}"
  echo "Location       : ${LOCATION}"
  echo "VNet           : ${VNET_NAME} (${VNET_ADDRESS_SPACE})"
  echo "Subnet         : ${SUBNET_DEFAULT_NAME} (${SUBNET_DEFAULT_PREFIX})"
  exit 0
fi

require_cmd az
ensure_logged_in

# --- 1. Resource group -------------------------------------------------------
echo "[1/3] Ensuring resource group ${RESOURCE_GROUP} in ${LOCATION}"
if ! az group show --name "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --output none
  echo "      created"
else
  echo "      already exists"
fi

# --- 2. VNet -----------------------------------------------------------------
echo "[2/3] Ensuring VNet ${VNET_NAME} (${VNET_ADDRESS_SPACE})"
if ! az network vnet show \
      --resource-group "${RESOURCE_GROUP}" \
      --name "${VNET_NAME}" >/dev/null 2>&1; then
  az network vnet create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VNET_NAME}" \
    --location "${LOCATION}" \
    --address-prefixes "${VNET_ADDRESS_SPACE}" \
    --output none
  echo "      created"
else
  echo "      already exists"
fi

# --- 3. Default subnet -------------------------------------------------------
echo "[3/3] Ensuring subnet ${SUBNET_DEFAULT_NAME} (${SUBNET_DEFAULT_PREFIX})"
if ! az network vnet subnet show \
      --resource-group "${RESOURCE_GROUP}" \
      --vnet-name "${VNET_NAME}" \
      --name "${SUBNET_DEFAULT_NAME}" >/dev/null 2>&1; then
  az network vnet subnet create \
    --resource-group "${RESOURCE_GROUP}" \
    --vnet-name "${VNET_NAME}" \
    --name "${SUBNET_DEFAULT_NAME}" \
    --address-prefixes "${SUBNET_DEFAULT_PREFIX}" \
    --output none
  echo "      created"
else
  echo "      already exists"
fi

# --- Summary -----------------------------------------------------------------
echo
echo "=== Summary ==="
echo "Resource group : ${RESOURCE_GROUP} (${LOCATION})"
echo "VNet           : ${VNET_NAME} (${VNET_ADDRESS_SPACE})"
echo "Subnet         : ${SUBNET_DEFAULT_NAME} (${SUBNET_DEFAULT_PREFIX})"
echo "Done. VNet is ready in ${RESOURCE_GROUP}."
