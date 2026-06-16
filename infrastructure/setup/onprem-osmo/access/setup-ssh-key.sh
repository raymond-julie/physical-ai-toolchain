#!/usr/bin/env bash
#
# setup-ssh-key.sh
#
# One-time setup that provisions an SSH keypair for connecting to an
# Azure Arc-enabled server via `az ssh arc` using a local user account.
#
# Steps performed:
#   1. Generate an ed25519 SSH keypair (skipped if it already exists).
#   2. Upload the public key to the target machine's
#      ~/.ssh/authorized_keys file for the specified local user.
#      The first invocation will prompt for the local user's password.
#   3. Verify a passwordless connection works using the new key.
#
# Re-running the script is safe: existing keys are reused and the
# public key is only appended once (idempotent grep check on the host).
#
# Identity values default to the Houston on-prem training cell and are
# overridable from the environment (SUBSCRIPTION, RESOURCE_GROUP, ARC_NAME,
# LOCAL_USER). RESOURCE_GROUP here is the resource group that holds the Arc
# machine (the facility RG).

set -euo pipefail

# --- Connection parameters ---------------------------------------------------
SUBSCRIPTION="${SUBSCRIPTION:-57b15bf0-e8dd-458a-9156-0694edd7ad4e}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-hou-facility}"
ARC_NAME="${ARC_NAME:-asus-nuc-4}"
LOCAL_USER="${LOCAL_USER:-edge}"

# --- Local key configuration -------------------------------------------------
KEY_DIR="${HOME}/.ssh"
KEY_NAME="arc_${ARC_NAME//-/_}_${LOCAL_USER}"
KEY_PATH="${SSH_KEY_PATH:-${KEY_DIR}/${KEY_NAME}}"
PUB_KEY_PATH="${KEY_PATH}.pub"

# --- 1. Generate keypair -----------------------------------------------------
mkdir -p "${KEY_DIR}"
chmod 700 "${KEY_DIR}"

if [[ -f "${KEY_PATH}" ]]; then
  echo "[1/3] Reusing existing key at ${KEY_PATH}"
else
  echo "[1/3] Generating new ed25519 keypair at ${KEY_PATH}"
  ssh-keygen \
    -t ed25519 \
    -f "${KEY_PATH}" \
    -N "" \
    -C "az-ssh-arc-${ARC_NAME}-${LOCAL_USER}"
fi

PUB_KEY_CONTENT="$(cat "${PUB_KEY_PATH}")"

# --- 2. Upload public key to the Arc machine ---------------------------------
# `az ssh arc -- <remote command>` opens an SSH session, runs the command,
# and exits. The first run will prompt for the local user's password.
echo "[2/3] Uploading public key to ${LOCAL_USER}@${ARC_NAME}"
echo "      (you will be prompted for the '${LOCAL_USER}' user's password)"

REMOTE_CMD="mkdir -p ~/.ssh && \
chmod 700 ~/.ssh && \
touch ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
grep -qxF '${PUB_KEY_CONTENT}' ~/.ssh/authorized_keys || \
echo '${PUB_KEY_CONTENT}' >> ~/.ssh/authorized_keys"

az ssh arc \
  --subscription "${SUBSCRIPTION}" \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ARC_NAME}" \
  --local-user "${LOCAL_USER}" \
  -- "${REMOTE_CMD}"

# --- 3. Verify key-based authentication --------------------------------------
echo "[3/3] Verifying key-based authentication"
az ssh arc \
  --subscription "${SUBSCRIPTION}" \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ARC_NAME}" \
  --local-user "${LOCAL_USER}" \
  --private-key-file "${KEY_PATH}" \
  -- "echo 'SSH key authentication succeeded for' \$(whoami)@\$(hostname)"

echo
echo "Done. Connect any time with:"
echo "  ./connect.sh"
