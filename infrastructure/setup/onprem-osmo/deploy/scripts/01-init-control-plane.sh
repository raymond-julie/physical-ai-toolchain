#!/usr/bin/env bash
#
# 01-init-control-plane.sh
# Initialize the Kubernetes control plane on the designated node.
#
# Run this script on the CONTROL PLANE node only, after running
# 00-prerequisites.sh on all nodes.
#
# Usage:
#   sudo bash scripts/01-init-control-plane.sh [--config path/to/inventory.env]

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

check_root

main() {
  local config_file=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        config_file="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done

  load_config "${config_file}"

  log_step "1" "Initializing Kubernetes control plane"

  init_cluster
  configure_kubectl
  install_cni
  maybe_untaint_control_plane
  # Generate the join command BEFORE any long-running steps so workers can
  # join even if later steps fail.
  generate_join_command
  ensure_helm
  # NOTE: We intentionally do NOT install a standalone ingress-nginx here.
  # The OSMO Helm chart (step 3) deploys its own quick-start ingress on
  # NodePort 30080. Installing a second controller here causes Helm to fail
  # with: 'Service "quick-start" is invalid: spec.ports[0].nodePort:
  #        Invalid value: 30080: provided port is already allocated'.
  # If you need a cluster-wide ingress for other workloads, install it on
  # different NodePorts (e.g. 30090/30453) AFTER step 3 succeeds.

  log_success "Control plane initialized successfully."
  log_info "Worker nodes can now join using the command saved in:"
  log_info "  $(get_project_dir)/config/join-command.sh"
}

##############################################################################
# Cluster Initialization
##############################################################################

init_cluster() {
  if [[ -f /etc/kubernetes/admin.conf ]]; then
    log_warning "Kubernetes cluster already initialized. Skipping kubeadm init."
    return
  fi

  log_info "Initializing Kubernetes cluster with kubeadm..."

  local advertise_address="${CONTROL_PLANE_IP}"

  kubeadm init \
    --apiserver-advertise-address="${advertise_address}" \
    --pod-network-cidr="${POD_NETWORK_CIDR}" \
    --service-cidr="${SERVICE_CIDR}" \
    --node-name="${CONTROL_PLANE_HOSTNAME}" \
    --cri-socket="unix:///run/containerd/containerd.sock"

  log_success "Kubernetes cluster initialized."
}

##############################################################################
# kubectl Configuration
##############################################################################

configure_kubectl() {
  log_info "Configuring kubectl for the current user..."

  local user_home
  user_home="$(eval echo "~${SUDO_USER:-root}")"
  local kube_dir="${user_home}/.kube"

  mkdir -p "${kube_dir}"
  cp /etc/kubernetes/admin.conf "${kube_dir}/config"

  if [[ -n "${SUDO_USER:-}" ]]; then
    chown -R "${SUDO_USER}:$(id -gn "${SUDO_USER}")" "${kube_dir}"
  fi

  # Also copy to project config for distribution
  local project_dir
  project_dir="$(get_project_dir)"
  cp /etc/kubernetes/admin.conf "${project_dir}/config/kubeconfig"
  chmod 0600 "${project_dir}/config/kubeconfig"

  if [[ -n "${SUDO_USER:-}" ]]; then
    chown "${SUDO_USER}:$(id -gn "${SUDO_USER}")" "${project_dir}/config/kubeconfig"
  fi

  log_success "kubectl configured. Kubeconfig saved to config/kubeconfig."
}

##############################################################################
# CNI Plugin (Calico)
##############################################################################

install_cni() {
  log_info "Installing Calico CNI plugin..."

  export KUBECONFIG=/etc/kubernetes/admin.conf

  kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/calico.yaml

  log_info "Waiting for Calico pods to become ready..."
  sleep 10
  kubectl wait --for=condition=Ready pods --all \
    --namespace kube-system --timeout=300s 2>/dev/null || true

  log_success "Calico CNI installed."
}

##############################################################################
# Control-plane taint (multi-node: honor UNTAINT_CONTROL_PLANE from inventory)
##############################################################################

maybe_untaint_control_plane() {
  if [[ "${UNTAINT_CONTROL_PLANE:-false}" != "true" && "${SINGLE_NODE:-false}" != "true" ]]; then
    return
  fi

  log_info "Untainting control plane so workloads can schedule on it..."
  export KUBECONFIG=/etc/kubernetes/admin.conf

  # kubectl returns non-zero if the taint is already absent; swallow that.
  kubectl taint nodes "${CONTROL_PLANE_HOSTNAME}" \
    node-role.kubernetes.io/control-plane:NoSchedule- 2>/dev/null || true
  kubectl taint nodes "${CONTROL_PLANE_HOSTNAME}" \
    node-role.kubernetes.io/master:NoSchedule- 2>/dev/null || true

  log_success "Control plane untainted."
}

##############################################################################
# Ingress NGINX
##############################################################################
# NOTE: Intentionally NOT installed here. OSMO's Helm chart ships its own
# ingress controller (quick-start) that owns NodePort 30080. A second
# ingress-nginx release at 30080 collides with it — see main() for details.

##############################################################################
# Helm (required by step 3)
##############################################################################

ensure_helm() {
  if command -v helm &>/dev/null; then
    log_info "Helm already installed: $(helm version --short 2>/dev/null || echo 'unknown')"
    return
  fi
  log_info "Installing Helm..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
  log_success "Helm installed: $(helm version --short 2>/dev/null || echo 'unknown')"
}

##############################################################################
# Join Command Generation
##############################################################################

generate_join_command() {
  log_info "Generating worker node join command..."

  export KUBECONFIG=/etc/kubernetes/admin.conf
  local project_dir
  project_dir="$(get_project_dir)"
  local join_file="${project_dir}/config/join-command.sh"

  kubeadm token create --print-join-command > "${join_file}"
  chmod 0600 "${join_file}"

  if [[ -n "${SUDO_USER:-}" ]]; then
    chown "${SUDO_USER}:$(id -gn "${SUDO_USER}")" "${join_file}"
  fi

  log_success "Join command saved to config/join-command.sh"
  log_info "Contents:"
  cat "${join_file}"
}

main "$@"
