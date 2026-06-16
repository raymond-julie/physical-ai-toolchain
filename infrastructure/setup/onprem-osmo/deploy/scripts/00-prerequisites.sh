#!/usr/bin/env bash
#
# 00-prerequisites.sh
# Install Kubernetes prerequisites on a node (control plane or worker).
#
# Run this script on EVERY machine that will join the OSMO cluster.
#
# Prerequisites: Ubuntu 22.04+ or equivalent Linux distribution
#
# Usage:
#   sudo bash scripts/00-prerequisites.sh

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

check_root

main() {
  log_step "0" "Installing prerequisites for OSMO Kubernetes cluster"

  install_system_packages
  disable_swap
  load_kernel_modules
  configure_sysctl
  install_containerd
  install_kubernetes_tools
  configure_crictl

  log_success "All prerequisites installed successfully."
  log_info "This node is ready to join a Kubernetes cluster."
}

##############################################################################
# System Packages
##############################################################################

install_system_packages() {
  log_info "Installing system packages..."

  apt-get update -qq
  apt-get install -y -qq \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    socat \
    conntrack \
    ipset \
    jq \
    bash-completion

  log_success "System packages installed."
}

##############################################################################
# Swap
##############################################################################

disable_swap() {
  log_info "Disabling swap (required by Kubernetes)..."

  swapoff -a || true
  sed -i '/\sswap\s/d' /etc/fstab

  log_success "Swap disabled."
}

##############################################################################
# Kernel Modules
##############################################################################

load_kernel_modules() {
  log_info "Loading required kernel modules..."

  cat > /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF

  modprobe overlay
  modprobe br_netfilter

  log_success "Kernel modules loaded."
}

##############################################################################
# Sysctl
##############################################################################

configure_sysctl() {
  log_info "Configuring sysctl parameters for Kubernetes networking..."

  cat > /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
fs.inotify.max_user_watches         = 1048576
fs.inotify.max_user_instances       = 512
EOF

  sysctl --system >/dev/null 2>&1

  log_success "Sysctl parameters configured."
}

##############################################################################
# Containerd
##############################################################################

install_containerd() {
  if command -v containerd &>/dev/null; then
    log_info "containerd already installed: $(containerd --version)"
    configure_containerd
    return
  fi

  log_info "Installing containerd..."

  # Add Docker repository for containerd
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  local arch
  arch="$(dpkg --print-architecture)"
  local codename
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"

  cat > /etc/apt/sources.list.d/docker.list <<EOF
deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${codename} stable
EOF

  apt-get update -qq
  apt-get install -y -qq containerd.io

  configure_containerd

  log_success "containerd installed and configured."
}

configure_containerd() {
  log_info "Configuring containerd with SystemdCgroup..."

  mkdir -p /etc/containerd
  containerd config default > /etc/containerd/config.toml

  # Enable SystemdCgroup (required by Kubernetes)
  sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml

  systemctl restart containerd
  systemctl enable containerd

  log_success "containerd configured."
}

##############################################################################
# Kubernetes Tools (kubeadm, kubelet, kubectl)
##############################################################################

install_kubernetes_tools() {
  local k8s_major_minor="${KUBERNETES_VERSION}"

  # If kubeadm is already present, verify it matches the pinned minor line.
  # Mixed minor versions across the cluster cause `kubeadm join` to fail with
  # messages like: "this version of kubeadm only supports deploying clusters
  # with the control plane version >= 1.34.0. Current version: v1.32.13".
  # We unconditionally re-install from the pinned apt repo in that case so
  # every node ends up on the same ${KUBERNETES_VERSION} line.
  if command -v kubeadm &>/dev/null; then
    local installed_ver
    installed_ver="$(kubeadm version -o short 2>/dev/null | sed 's/^v//' || echo 'unknown')"
    local installed_minor="${installed_ver%.*}"   # e.g. "1.35.3" -> "1.35"

    if [[ "${installed_minor}" == "${k8s_major_minor}" ]]; then
      log_info "Kubernetes tools already installed at v${installed_ver} (matches pinned v${k8s_major_minor}.x)"
      return
    fi

    log_warning "Kubernetes tools installed at v${installed_ver}, but cluster is pinned to v${k8s_major_minor}.x. Re-installing..."
    apt-mark unhold kubelet kubeadm kubectl 2>/dev/null || true
    # Remove the old apt source so we don't fight with a stale pkgs.k8s.io line
    rm -f /etc/apt/sources.list.d/kubernetes.list
  else
    log_info "Installing kubeadm, kubelet, and kubectl..."
  fi

  install -m 0755 -d /etc/apt/keyrings

  # Re-write the keyring each run; older keys may point at a different minor.
  curl -fsSL "https://pkgs.k8s.io/core:/stable:/v${k8s_major_minor}/deb/Release.key" \
    | gpg --batch --yes --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

  cat > /etc/apt/sources.list.d/kubernetes.list <<EOF
deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v${k8s_major_minor}/deb/ /
EOF

  apt-get update -qq
  # Resolve the latest package version available on the pinned minor line
  # (e.g. "1.32.13-1.1") so apt will downgrade from a newer minor like 1.35.
  local pin_ver
  pin_ver="$(apt-cache madison kubeadm \
    | awk -v m="${k8s_major_minor}" '$3 ~ ("^" m ".") {print $3; exit}')"
  if [[ -z "${pin_ver}" ]]; then
    log_warning "Could not resolve a ${k8s_major_minor}.x package version from apt; installing unpinned."
    apt-get install -y -qq --allow-downgrades --allow-change-held-packages \
      kubelet kubeadm kubectl
  else
    log_info "Pinning kubelet/kubeadm/kubectl to ${pin_ver}"
    apt-get install -y -qq --allow-downgrades --allow-change-held-packages \
      "kubelet=${pin_ver}" "kubeadm=${pin_ver}" "kubectl=${pin_ver}"
  fi
  apt-mark hold kubelet kubeadm kubectl

  systemctl enable kubelet

  log_success "Kubernetes ${k8s_major_minor} tools installed ($(kubeadm version -o short 2>/dev/null || echo '?'))."
}

##############################################################################
# CRI-CTL Configuration
##############################################################################

configure_crictl() {
  log_info "Configuring crictl..."

  cat > /etc/crictl.yaml <<EOF
runtime-endpoint: unix:///run/containerd/containerd.sock
image-endpoint: unix:///run/containerd/containerd.sock
timeout: 10
EOF

  log_success "crictl configured."
}

main "$@"
