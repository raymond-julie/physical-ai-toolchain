#!/usr/bin/env bash
#
# connect.sh
#
# Opens an interactive SSH session to the Azure Arc-enabled server
# using the keypair provisioned by setup-ssh-key.sh.
#
# Identity values default to the Houston on-prem training cell and are
# overridable from the environment (SUBSCRIPTION, RESOURCE_GROUP, ARC_NAME,
# LOCAL_USER). RESOURCE_GROUP here is the resource group that holds the Arc
# machine (the facility RG), which differs from the training-cell RG used by
# 01-create-vnet.sh.

set -euo pipefail

SUBSCRIPTION="${SUBSCRIPTION:-57b15bf0-e8dd-458a-9156-0694edd7ad4e}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-hou-facility}"
ARC_NAME="${ARC_NAME:-asus-nuc-4}"
LOCAL_USER="${LOCAL_USER:-edge}"

KEY_PATH="${SSH_KEY_PATH:-${HOME}/.ssh/arc_${ARC_NAME//-/_}_${LOCAL_USER}}"

if [[ ! -f "${KEY_PATH}" ]]; then
  echo "Private key not found at ${KEY_PATH}." >&2
  echo "Run ./setup-ssh-key.sh first." >&2
  exit 1
fi

# Forward any extra args after `--` so this can be used non-interactively:
#   ./connect.sh                         # interactive shell
#   ./connect.sh -- 'echo hi && uname'   # one-shot remote command
exec az ssh arc \
  --subscription "${SUBSCRIPTION}" \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ARC_NAME}" \
  --local-user "${LOCAL_USER}" \
  --private-key-file "${KEY_PATH}" \
  "$@"
