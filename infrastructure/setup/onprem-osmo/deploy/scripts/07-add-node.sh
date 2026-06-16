#!/usr/bin/env bash
#
# 07-add-node.sh
# Add a new worker node to an existing OSMO Kubernetes cluster.
#
# Run this script from the control plane (or any machine with kubectl access
# and SSH connectivity to the new node).
#
# The script will:
#   1. SSH to the new node and install prerequisites
#   2. Generate a fresh join token
#   3. SSH to the new node and join it to the cluster
#   4. Apply Kubernetes labels
#
# Usage:
#   bash scripts/07-add-node.sh --node-ip 192.168.1.104 --node-user ubuntu [OPTIONS]
#
# Options:
#   --node-ip IP          IP address of the new node (required)
#   --node-user USER      SSH user on the new node (default: ubuntu)
#   --node-hostname NAME  Hostname for the node (default: auto-generated)
#   --labels LABELS       Kubernetes labels (default: node_group=compute)
#   --ssh-key PATH        SSH private key path (default: ~/.ssh/id_rsa)
#   --config PATH         Inventory config file (to append the new node)
#   --gpu                 Install GPU Operator on this node
#   --skip-prereqs        Skip prerequisite installation
#   --help, -h            Show this help message

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

main() {
  local node_ip=""
  local node_user="ubuntu"
  local node_hostname=""
  local node_labels="node_group=compute"
  local ssh_key="${HOME}/.ssh/id_rsa"
  local config_file=""
  local enable_gpu=false
  local skip_prereqs=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --node-ip)
        node_ip="$2"
        shift 2
        ;;
      --node-user)
        node_user="$2"
        shift 2
        ;;
      --node-hostname)
        node_hostname="$2"
        shift 2
        ;;
      --labels)
        node_labels="$2"
        shift 2
        ;;
      --ssh-key)
        ssh_key="$2"
        shift 2
        ;;
      --config)
        config_file="$2"
        shift 2
        ;;
      --gpu)
        enable_gpu=true
        shift
        ;;
      --skip-prereqs)
        skip_prereqs=true
        shift
        ;;
      --help|-h)
        usage
        ;;
      *)
        log_error "Unknown option: $1"
        usage
        ;;
    esac
  done

  # Validate required parameters
  if [[ -z "${node_ip}" ]]; then
    log_error "--node-ip is required."
    usage
  fi

  if ! validate_ip "${node_ip}"; then
    log_error "Invalid IP address: ${node_ip}"
    exit 1
  fi

  # Auto-generate hostname if not provided
  if [[ -z "${node_hostname}" ]]; then
    local existing_count
    existing_count="$(kubectl get nodes --no-headers 2>/dev/null | wc -l || echo 0)"
    node_hostname="osmo-node-$((existing_count + 1))"
    log_info "Auto-generated hostname: ${node_hostname}"
  fi

  # Verify kubectl access
  check_command "kubectl"
  if ! kubectl cluster-info &>/dev/null; then
    log_error "Cannot connect to Kubernetes cluster. Ensure kubectl is configured."
    exit 1
  fi

  log_step "0" "Adding worker node: ${node_ip} (${node_hostname})"

  echo "  Node IP:       ${node_ip}"
  echo "  Node User:     ${node_user}"
  echo "  Node Hostname: ${node_hostname}"
  echo "  Labels:        ${node_labels}"
  echo "  GPU:           ${enable_gpu}"
  echo ""

  # Check SSH connectivity
  verify_ssh "${node_ip}" "${node_user}" "${ssh_key}"

  #--- Phase 1: Install prerequisites on the new node ---
  if [[ "${skip_prereqs}" == "false" ]]; then
    log_step "1" "Installing prerequisites on ${node_ip}"
    copy_scripts_to_node "${node_ip}" "${node_user}" "${ssh_key}"
    run_remote "${node_ip}" "${node_user}" "${ssh_key}" \
      "sudo bash ~/osmo-deployment/scripts/00-prerequisites.sh"
  else
    log_info "Skipping prerequisites (--skip-prereqs)."
  fi

  #--- Phase 2: Generate join token and join ---
  log_step "2" "Joining node to the cluster"
  local join_command
  join_command="$(generate_join_token)"
  log_info "Join token generated."

  run_remote "${node_ip}" "${node_user}" "${ssh_key}" \
    "sudo ${join_command} --cri-socket='unix:///run/containerd/containerd.sock'"

  #--- Phase 3: Wait for node and apply labels ---
  log_step "3" "Configuring node in the cluster"
  wait_for_new_node "${node_ip}" "${node_hostname}"
  apply_node_labels "${node_hostname}" "${node_labels}"

  #--- Phase 4: GPU Operator (if needed) ---
  if [[ "${enable_gpu}" == "true" ]]; then
    log_step "4" "Enabling GPU on ${node_hostname}"
    # The GPU Operator is cluster-wide; just needs to be installed once
    ensure_gpu_operator
  fi

  #--- Phase 5: Update inventory config ---
  if [[ -n "${config_file}" ]]; then
    append_to_inventory "${config_file}" "${node_ip}" "${node_hostname}" "${node_user}" "${node_labels}"
  else
    log_info "Tip: To persist this node in your config, add to inventory.env:"
    log_info "  WORKER_IPS+=( \"${node_ip}\" )"
    log_info "  WORKER_HOSTNAMES+=( \"${node_hostname}\" )"
    log_info "  WORKER_USERS+=( \"${node_user}\" )"
    log_info "  WORKER_LABELS+=( \"${node_labels}\" )"
  fi

  #--- Done ---
  echo ""
  log_success "Node '${node_hostname}' (${node_ip}) added to the cluster!"
  echo ""
  kubectl get nodes -o wide
}

usage() {
  cat <<EOF
Usage: ${0##*/} --node-ip <IP> [OPTIONS]

Add a new worker node to an existing OSMO Kubernetes cluster.

Required:
  --node-ip IP          IP address of the new node

Options:
  --node-user USER      SSH user (default: ubuntu)
  --node-hostname NAME  Hostname (default: auto-generated)
  --labels LABELS       Kubernetes labels (default: node_group=compute)
  --ssh-key PATH        SSH key path (default: ~/.ssh/id_rsa)
  --config PATH         Inventory file to append the new node to
  --gpu                 Ensure GPU Operator is installed
  --skip-prereqs        Skip prerequisite installation on the new node
  --help, -h            Show this help

Examples:
  # Add a compute worker
  bash scripts/07-add-node.sh --node-ip 192.168.1.104 --node-user ubuntu

  # Add a GPU worker with custom hostname
  bash scripts/07-add-node.sh --node-ip 192.168.1.105 --node-user ubuntu \\
    --node-hostname gpu-worker-1 --labels "node_group=compute" --gpu

  # Add and persist to inventory config
  bash scripts/07-add-node.sh --node-ip 192.168.1.104 --node-user ubuntu \\
    --config config/inventory.env
EOF
  exit 1
}

##############################################################################
# SSH Helpers
##############################################################################

verify_ssh() {
  local ip="$1"
  local user="$2"
  local key="$3"

  log_info "Verifying SSH connectivity to ${user}@${ip}..."

  if ! ssh -i "${key}" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    "${user}@${ip}" "echo 'SSH OK'" &>/dev/null; then
    log_error "Cannot SSH to ${user}@${ip}. Check:"
    log_info "  - The node is reachable: ping ${ip}"
    log_info "  - SSH key is authorized: ssh-copy-id -i ${key} ${user}@${ip}"
    exit 1
  fi

  log_success "SSH connection verified."
}

run_remote() {
  local ip="$1"
  local user="$2"
  local key="$3"
  local cmd="$4"

  log_info "[SSH] ${user}@${ip}: ${cmd}"

  ssh -i "${key}" \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=10 \
    "${user}@${ip}" \
    "${cmd}" 2>&1
}

copy_scripts_to_node() {
  local ip="$1"
  local user="$2"
  local key="$3"

  log_info "Copying deployment scripts to ${user}@${ip}..."

  local project_dir
  project_dir="$(get_project_dir)"

  # Create remote directory
  run_remote "${ip}" "${user}" "${key}" \
    "mkdir -p ~/osmo-deployment/scripts ~/osmo-deployment/config"

  # Copy scripts
  scp -i "${key}" -o StrictHostKeyChecking=no -r \
    "${project_dir}/scripts/." \
    "${user}@${ip}:~/osmo-deployment/scripts/" 2>&1

  # Copy config
  scp -i "${key}" -o StrictHostKeyChecking=no -r \
    "${project_dir}/config/." \
    "${user}@${ip}:~/osmo-deployment/config/" 2>&1

  # Make executable
  run_remote "${ip}" "${user}" "${key}" \
    "chmod +x ~/osmo-deployment/scripts/*.sh"

  log_success "Scripts copied to ${ip}."
}

##############################################################################
# Cluster Operations
##############################################################################

generate_join_token() {
  # Generate a new token (lasts 24h by default)
  kubeadm token create --print-join-command 2>/dev/null
}

wait_for_new_node() {
  local ip="$1"
  local hostname="$2"

  log_info "Waiting for node to appear in the cluster..."

  local attempts=0
  local max_attempts=60  # 5 minutes (5s intervals)

  while [[ ${attempts} -lt ${max_attempts} ]]; do
    # Check by hostname first, fall back to checking by IP
    if kubectl get node "${hostname}" &>/dev/null; then
      log_success "Node '${hostname}' found in cluster."
      break
    fi

    # Check if any new NotReady/Ready node matches the IP
    if kubectl get nodes -o wide --no-headers 2>/dev/null | grep -q "${ip}"; then
      log_success "Node with IP ${ip} found in cluster."
      break
    fi

    attempts=$((attempts + 1))
    sleep 5
  done

  if [[ ${attempts} -ge ${max_attempts} ]]; then
    log_warning "Timed out waiting for node. It may still be joining."
    kubectl get nodes -o wide || true
    return
  fi

  # Wait for the node to be Ready
  log_info "Waiting for node to become Ready..."
  kubectl wait --for=condition=Ready "node/${hostname}" --timeout=300s 2>/dev/null || {
    log_warning "Node may not be fully ready yet."
    kubectl get nodes -o wide || true
  }
}

apply_node_labels() {
  local hostname="$1"
  local labels="$2"

  log_info "Applying labels to node '${hostname}': ${labels}"

  # Split space or comma-separated labels and apply each
  local IFS=', '
  for label in ${labels}; do
    kubectl label node "${hostname}" "${label}" --overwrite 2>/dev/null || true
  done

  log_success "Labels applied."
}

ensure_gpu_operator() {
  if helm list -n gpu-operator 2>/dev/null | grep -q gpu-operator; then
    log_info "GPU Operator already installed. The new node will be detected automatically."
    return
  fi

  log_info "Installing NVIDIA GPU Operator (${GPU_OPERATOR_VERSION})..."

  if ! command -v helm &>/dev/null; then
    log_error "Helm is required. Install Helm first."
    return
  fi

  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia 2>/dev/null || true
  helm repo update

  helm upgrade --install gpu-operator nvidia/gpu-operator \
    --namespace gpu-operator \
    --create-namespace \
    --set driver.enabled=true \
    --set toolkit.enabled=true \
    --set nfd.enabled=true \
    --wait --timeout 10m

  log_success "GPU Operator installed."
}

##############################################################################
# Inventory Update
##############################################################################

append_to_inventory() {
  local config_file="$1"
  local ip="$2"
  local hostname="$3"
  local user="$4"
  local labels="$5"

  if [[ ! -f "${config_file}" ]]; then
    log_warning "Config file not found: ${config_file}. Skipping inventory update."
    return
  fi

  log_info "Appending new node to ${config_file}..."

  # This is a best-effort append. For complex configs, manual editing is safer.
  # We add a comment marking the addition.
  cat >> "${config_file}" <<EOF

# Node added on $(date '+%Y-%m-%d %H:%M:%S')
# To include in arrays, update WORKER_IPS, WORKER_HOSTNAMES, WORKER_USERS, WORKER_LABELS above.
# New node: ${hostname} (${ip}) user=${user} labels=${labels}
EOF

  log_info "Note: Array values (WORKER_IPS, etc.) must be updated manually in ${config_file}."
  log_info "Add the following entries to the existing arrays:"
  log_info "  WORKER_IPS+=( \"${ip}\" )"
  log_info "  WORKER_HOSTNAMES+=( \"${hostname}\" )"
  log_info "  WORKER_USERS+=( \"${user}\" )"
  log_info "  WORKER_LABELS+=( \"${labels}\" )"
}

main "$@"
