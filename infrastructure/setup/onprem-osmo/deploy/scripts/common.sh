#!/usr/bin/env bash
#
# common.sh
# Shared utility functions for OSMO cluster deployment scripts

# shellcheck disable=SC2034  # Variables used by sourced scripts

set -euo pipefail

##############################################################################
# Colors and Formatting
##############################################################################

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

##############################################################################
# Logging Functions
##############################################################################

log_info() {
  printf "${BLUE}[INFO]${NC} %s\n" "$1"
}

log_success() {
  printf "${GREEN}[OK]${NC} %s\n" "$1"
}

log_warning() {
  printf "${YELLOW}[WARN]${NC} %s\n" "$1" >&2
}

log_error() {
  printf "${RED}[ERROR]${NC} %s\n" "$1" >&2
}

log_step() {
  local step_num="$1"
  local step_msg="$2"
  printf "\n${CYAN}=== Step %s: %s ===${NC}\n\n" "${step_num}" "${step_msg}"
}

##############################################################################
# Command Validation
##############################################################################

check_command() {
  local cmd="$1"
  if ! command -v "${cmd}" &>/dev/null; then
    log_error "'${cmd}' is required but not installed."
    return 1
  fi
}

check_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    log_error "This script must be run as root or with sudo."
    exit 1
  fi
}

##############################################################################
# Configuration Loading
##############################################################################

load_config() {
  local config_file="${1:-}"
  if [[ -z "${config_file}" ]]; then
    config_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/inventory.env"
  fi

  if [[ ! -f "${config_file}" ]]; then
    log_error "Configuration file not found: ${config_file}"
    log_info "Copy config/inventory.example.env to config/inventory.env and edit it."
    exit 1
  fi

  # shellcheck source=/dev/null
  source "${config_file}"
  log_success "Configuration loaded from ${config_file}"
}

##############################################################################
# Wait Utilities
##############################################################################

wait_for_pods() {
  local namespace="$1"
  local timeout="${2:-300}"
  local label="${3:-}"

  log_info "Waiting for pods in namespace '${namespace}' to be ready (timeout: ${timeout}s)..."

  local cmd=("kubectl" "wait" "--for=condition=Ready" "pods" "--all"
    "--namespace" "${namespace}" "--timeout=${timeout}s")

  if [[ -n "${label}" ]]; then
    cmd+=("-l" "${label}")
  fi

  if "${cmd[@]}" 2>/dev/null; then
    log_success "All pods in '${namespace}' are ready."
  else
    log_warning "Some pods in '${namespace}' may not be ready. Checking status..."
    kubectl get pods --namespace "${namespace}" 2>/dev/null || true
  fi
}

wait_for_nodes() {
  local timeout="${1:-300}"

  log_info "Waiting for all nodes to be ready (timeout: ${timeout}s)..."
  if kubectl wait --for=condition=Ready nodes --all --timeout="${timeout}s" 2>/dev/null; then
    log_success "All nodes are ready."
  else
    log_warning "Some nodes may not be ready. Checking status..."
    kubectl get nodes 2>/dev/null || true
  fi
}

##############################################################################
# Validation
##############################################################################

validate_ip() {
  local ip="$1"
  if [[ "${ip}" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
    return 0
  fi
  return 1
}

validate_hostname() {
  local host="$1"
  if [[ "${host}" =~ ^[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?$ ]]; then
    return 0
  fi
  return 1
}

##############################################################################
# Version Defaults
##############################################################################

KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.32}"
CONTAINERD_VERSION="${CONTAINERD_VERSION:-1.7}"
KAI_SCHEDULER_VERSION="${KAI_SCHEDULER_VERSION:-v0.8.1}"
GPU_OPERATOR_VERSION="${GPU_OPERATOR_VERSION:-v25.10.0}"
OSMO_NAMESPACE="${OSMO_NAMESPACE:-osmo}"
OSMO_HOSTNAME="${OSMO_HOSTNAME:-quick-start.osmo}"
SINGLE_NODE="${SINGLE_NODE:-false}"
UNTAINT_CONTROL_PLANE="${UNTAINT_CONTROL_PLANE:-false}"
POD_NETWORK_CIDR="${POD_NETWORK_CIDR:-10.244.0.0/16}"
SERVICE_CIDR="${SERVICE_CIDR:-10.96.0.0/12}"

##############################################################################
# Script Directory
##############################################################################

get_script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

get_project_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}
