#!/usr/bin/env bash
#
# 02-join-worker.sh
# Join a worker node to the Kubernetes cluster.
#
# Run this script on EACH WORKER node after the control plane is initialized.
#
# Usage:
#   sudo bash scripts/02-join-worker.sh --join-command "kubeadm join ..." [--labels "key=value,key2=value2"]
#   sudo bash scripts/02-join-worker.sh --join-file path/to/join-command.sh [--labels "key=value"]

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

check_root

main() {
  local join_command=""
  local join_file=""
  local node_labels=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --join-command)
        join_command="$2"
        shift 2
        ;;
      --join-file)
        join_file="$2"
        shift 2
        ;;
      --labels)
        node_labels="$2"
        shift 2
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

  # Resolve join command
  if [[ -z "${join_command}" ]]; then
    if [[ -n "${join_file}" && -f "${join_file}" ]]; then
      join_command="$(cat "${join_file}")"
    else
      # Try default location
      local default_join="$(get_project_dir)/config/join-command.sh"
      if [[ -f "${default_join}" ]]; then
        join_command="$(cat "${default_join}")"
      else
        log_error "No join command provided. Use --join-command or --join-file."
        usage
      fi
    fi
  fi

  log_step "2" "Joining worker node to the Kubernetes cluster"

  join_cluster "${join_command}"
  apply_labels "${node_labels}"

  log_success "Worker node joined the cluster successfully."
}

usage() {
  cat <<EOF
Usage: ${0##*/} [OPTIONS]

Options:
  --join-command CMD   The kubeadm join command string
  --join-file PATH     Path to file containing the join command
  --labels LABELS      Comma-separated node labels (e.g., node_group=compute)
  --help, -h           Show this help message

If neither --join-command nor --join-file is provided, the script looks for
config/join-command.sh relative to the project root.
EOF
  exit 1
}

##############################################################################
# Join Cluster
##############################################################################

join_cluster() {
  local cmd="$1"

  if [[ -f /etc/kubernetes/kubelet.conf ]]; then
    log_warning "This node appears to already be part of a cluster. Skipping join."
    return
  fi

  log_info "Joining cluster..."
  eval "${cmd}" --cri-socket="unix:///run/containerd/containerd.sock"

  log_success "Node joined the cluster."
}

##############################################################################
# Apply Labels
##############################################################################

apply_labels() {
  local labels="$1"

  if [[ -z "${labels}" ]]; then
    log_info "No labels specified. Node will use default labels."
    return
  fi

  local node_name
  node_name="$(hostname)"

  log_info "Applying labels to node '${node_name}': ${labels}"

  # Labels are applied from the control plane. Save label instructions.
  local project_dir
  project_dir="$(get_project_dir)"
  local label_file="${project_dir}/config/pending-labels.txt"

  echo "kubectl label node ${node_name} ${labels} --overwrite" >> "${label_file}"

  log_info "Label command saved to config/pending-labels.txt."
  log_info "Run the label commands from the control plane after all nodes join."
}

main "$@"
