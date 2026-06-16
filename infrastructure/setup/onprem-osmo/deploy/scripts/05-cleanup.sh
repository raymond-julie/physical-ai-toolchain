#!/usr/bin/env bash
#
# 05-cleanup.sh
# Tear down the OSMO cluster and clean up all resources.
#
# Run this script on the CONTROL PLANE node.
#
# Usage:
#   sudo bash scripts/05-cleanup.sh [--config path/to/inventory.env] [--keep-k8s]

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

check_root

main() {
  local config_file=""
  local keep_k8s=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        config_file="$2"
        shift 2
        ;;
      --keep-k8s)
        keep_k8s=true
        shift
        ;;
      --help|-h)
        usage
        ;;
      *)
        shift
        ;;
    esac
  done

  load_config "${config_file}"

  log_step "5" "Cleaning up OSMO cluster"

  remove_osmo
  remove_kai_scheduler
  remove_gpu_operator

  if [[ "${keep_k8s}" == "false" ]]; then
    reset_kubernetes
  else
    log_info "Kubernetes cluster preserved (--keep-k8s)."
  fi

  cleanup_config_files

  log_success "Cleanup complete."
}

usage() {
  cat <<EOF
Usage: ${0##*/} [OPTIONS]

Options:
  --config PATH    Path to inventory.env configuration file
  --keep-k8s       Remove OSMO but keep the Kubernetes cluster intact
  --help, -h       Show this help message
EOF
  exit 1
}

##############################################################################
# OSMO Removal
##############################################################################

remove_osmo() {
  log_info "Removing OSMO deployment..."

  export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"

  if helm list -n "${OSMO_NAMESPACE}" 2>/dev/null | grep -q osmo; then
    helm uninstall osmo --namespace "${OSMO_NAMESPACE}" --wait --timeout 5m 2>/dev/null || true
    log_success "OSMO Helm release removed."
  else
    log_info "No OSMO Helm release found."
  fi

  # Jobs have immutable pod templates; delete them explicitly so a later
  # redeploy doesn't hit "field is immutable" errors.
  kubectl delete jobs --all -n "${OSMO_NAMESPACE}" --ignore-not-found 2>/dev/null || true

  # Legacy standalone ingress-nginx release (older versions of
  # 01-init-control-plane.sh installed it on NodePort 30080, which now
  # collides with OSMO's quick-start ingress).
  if helm list -n ingress-nginx 2>/dev/null | grep -q ingress-nginx; then
    log_info "Removing legacy ingress-nginx release..."
    helm uninstall ingress-nginx -n ingress-nginx --wait 2>/dev/null || true
    kubectl delete namespace ingress-nginx --timeout=60s 2>/dev/null || true
  fi

  # Delete namespace
  kubectl delete namespace "${OSMO_NAMESPACE}" --timeout=60s 2>/dev/null || true
}

##############################################################################
# KAI Scheduler Removal
##############################################################################

remove_kai_scheduler() {
  log_info "Removing KAI Scheduler..."

  if helm list -n kai-scheduler 2>/dev/null | grep -q kai-scheduler; then
    helm uninstall kai-scheduler --namespace kai-scheduler --wait 2>/dev/null || true
    kubectl delete namespace kai-scheduler --timeout=60s 2>/dev/null || true
    log_success "KAI Scheduler removed."
  else
    log_info "No KAI Scheduler found."
  fi
}

##############################################################################
# GPU Operator Removal
##############################################################################

remove_gpu_operator() {
  log_info "Removing GPU Operator..."

  if helm list -n gpu-operator 2>/dev/null | grep -q gpu-operator; then
    helm uninstall gpu-operator --namespace gpu-operator --wait 2>/dev/null || true
    kubectl delete namespace gpu-operator --timeout=60s 2>/dev/null || true
    log_success "GPU Operator removed."
  else
    log_info "No GPU Operator found."
  fi
}

##############################################################################
# Kubernetes Reset
##############################################################################

reset_kubernetes() {
  log_info "Resetting Kubernetes cluster..."

  if command -v kubeadm &>/dev/null; then
    kubeadm reset -f 2>/dev/null || true
  fi

  # Kill any leftover pods/containers kubeadm reset didn't clean up.
  if command -v crictl &>/dev/null; then
    crictl rm -af 2>/dev/null || true
    crictl rmp -af 2>/dev/null || true
  fi

  # Any sandboxes that couldn't be removed (e.g. because CNI is already gone)
  # are cleared by wiping containerd's CRI metadata. Safe: all pods are down.
  if systemctl is-active containerd &>/dev/null; then
    systemctl stop containerd 2>/dev/null || true
    rm -rf /var/lib/containerd/io.containerd.metadata.v1.bolt \
           /var/lib/containerd/io.containerd.runtime.v2.task \
           /var/lib/containerd/io.containerd.sandbox.controller.v1.shim \
           /run/containerd/io.containerd.runtime.v2.task \
           /run/containerd/io.containerd.sandbox.controller.v1.shim 2>/dev/null || true
    systemctl start containerd 2>/dev/null || true
  fi

  # Stop transient kubepods*.slice systemd units (kubeadm reset leaves these
  # active because their libcontainer cgroups are still mounted).
  for s in $(systemctl list-units --all --no-legend 2>/dev/null | awk '/kubepods/ {print $1}'); do
    systemctl stop "$s" 2>/dev/null || true
    systemctl reset-failed "$s" 2>/dev/null || true
  done

  # Destroy Calico ipsets (would otherwise conflict with a fresh Calico install).
  if command -v ipset &>/dev/null; then
    for s in $(ipset list -n 2>/dev/null | grep -E '^cali'); do
      ipset destroy "$s" 2>/dev/null || true
    done
  fi

  # Delete Calico BGP-learned (bird) routes, including pod CIDR blackhole routes.
  for r in $(ip route 2>/dev/null | awk '/proto bird/ {print $1" "$2}'); do
    ip route del $r 2>/dev/null || true
  done

  # Calico uses ip rule priority 220 + routing table 220 for BGP nexthops.
  ip rule del priority 220 2>/dev/null || true
  ip route flush table 220 2>/dev/null || true

  # Remove stale /etc/hosts entries added by earlier deploys.
  sed -i '/quick-start\.osmo/d; /\.cluster\.local/d' /etc/hosts 2>/dev/null || true

  # Clean up networking state
  rm -rf /etc/cni/net.d /var/lib/cni /var/run/kubernetes 2>/dev/null || true

  # Calico state (installed by 01-init-control-plane.sh when CNI=calico).
  # Calico bind-mounts a cgroup into /run/calico, so unmount first.
  for mp in $(mount | awk '/\/run\/calico/ {print $3}'); do
    umount "$mp" 2>/dev/null || umount -l "$mp" 2>/dev/null || true
  done
  rm -rf /var/lib/calico /var/log/calico /run/calico /run/flannel 2>/dev/null || true

  # Pod/container logs
  rm -rf /var/log/pods /var/log/containers 2>/dev/null || true

  # Delete Calico/CNI virtual interfaces so they don't linger until reboot.
  for iface in $(ip -br link show 2>/dev/null | awk '/^cali/ {print $1}' | cut -d@ -f1); do
    ip link del "$iface" 2>/dev/null || true
  done
  # Orphaned veth peers from stopped pods (kubeadm reset leaves these when
  # CNI was already removed).
  for v in $(ip -br link show type veth 2>/dev/null | awk '{print $1}' | cut -d@ -f1 | grep -vE '^(docker|br-)'); do
    ip link del "$v" 2>/dev/null || true
  done
  for iface in tunl0 vxlan.calico kube-ipvs0 cni0 flannel.1 dummy0; do
    ip link del "$iface" 2>/dev/null || true
  done
  # Delete stale CNI network namespaces (pod sandboxes' netns)
  for ns in $(ip netns list 2>/dev/null | awk '/^cni-/ {print $1}'); do
    ip netns del "$ns" 2>/dev/null || true
  done
  # Unload the ipip module so tunl0 doesn't auto-recreate (reloaded on next init).
  modprobe -r ipip 2>/dev/null || true

  # Flush AND delete user-defined chains in every iptables table. kube-proxy
  # creates KUBE-* chains in filter/nat/mangle; -F alone only empties them.
  for t in filter nat mangle raw; do
    iptables -t "$t" -F 2>/dev/null || true
    iptables -t "$t" -X 2>/dev/null || true
  done
  ipvsadm --clear 2>/dev/null || true

  # Remove kubelet/etcd state so a fresh init doesn't hit "directory not empty"
  rm -rf /var/lib/kubelet /var/lib/etcd /etc/kubernetes 2>/dev/null || true

  # Remove kube config
  rm -rf "${HOME}/.kube" 2>/dev/null || true
  if [[ -n "${SUDO_USER:-}" ]]; then
    rm -rf "/home/${SUDO_USER}/.kube" "/home/${SUDO_USER}/.cache/helm" \
           "/home/${SUDO_USER}/.config/helm" "/home/${SUDO_USER}/.local/share/helm" 2>/dev/null || true
  fi

  log_success "Kubernetes cluster reset."
  log_info "Run 'kubeadm reset' on each worker node as well."
}

##############################################################################
# Configuration Cleanup
##############################################################################

cleanup_config_files() {
  log_info "Cleaning up generated configuration files..."

  local project_dir
  project_dir="$(get_project_dir)"

  rm -f "${project_dir}/config/join-command.sh" 2>/dev/null || true
  rm -f "${project_dir}/config/kubeconfig" 2>/dev/null || true
  rm -f "${project_dir}/config/pending-labels.txt" 2>/dev/null || true

  log_success "Configuration files cleaned up."
}

main "$@"
