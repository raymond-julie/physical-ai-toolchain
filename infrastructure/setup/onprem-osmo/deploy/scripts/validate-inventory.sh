#!/usr/bin/env bash
# Non-destructive validation: source common.sh + inventory under `set -u`
# and print every variable referenced by deploy scripts. Any unbound
# variable will cause bash to abort with a clear error.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

CONFIG="${1:-${SCRIPT_DIR}/../config/inventory.env}"
load_config "${CONFIG}"

echo "--- Scalars ---"
echo "CONTROL_PLANE_IP=${CONTROL_PLANE_IP}"
echo "CONTROL_PLANE_HOSTNAME=${CONTROL_PLANE_HOSTNAME}"
echo "CONTROL_PLANE_USER=${CONTROL_PLANE_USER}"
echo "ENABLE_GPU=${ENABLE_GPU}"
echo "UNTAINT_CONTROL_PLANE=${UNTAINT_CONTROL_PLANE}"
echo "SINGLE_NODE=${SINGLE_NODE}"
echo "POD_NETWORK_CIDR=${POD_NETWORK_CIDR}"
echo "SERVICE_CIDR=${SERVICE_CIDR}"
echo "KUBERNETES_VERSION=${KUBERNETES_VERSION}"
echo "KAI_SCHEDULER_VERSION=${KAI_SCHEDULER_VERSION}"
echo "GPU_OPERATOR_VERSION=${GPU_OPERATOR_VERSION}"
echo "OSMO_NAMESPACE=${OSMO_NAMESPACE}"
echo "OSMO_HOSTNAME=${OSMO_HOSTNAME}"

echo "--- Arrays ---"
echo "WORKER_IPS=(${WORKER_IPS[*]})"
echo "WORKER_HOSTNAMES=(${WORKER_HOSTNAMES[*]})"
echo "WORKER_USERS=(${WORKER_USERS[*]})"
echo "WORKER_LABELS=(${WORKER_LABELS[*]})"

# GPU worker indices may be unset when ENABLE_GPU=false
if [[ "${ENABLE_GPU}" == "true" ]]; then
  echo "GPU_WORKER_INDICES=(${GPU_WORKER_INDICES[*]})"
fi

echo "ALL_VARS_OK"
