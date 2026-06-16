#!/usr/bin/env bash
#
# 03-deploy-osmo.sh
# Deploy NVIDIA OSMO on the Kubernetes cluster.
#
# Run this script on the CONTROL PLANE node after all worker nodes have joined.
#
# This script installs:
#   1. Node labels for OSMO scheduling
#   2. KAI Scheduler (workload co-scheduling)
#   3. NVIDIA GPU Operator (optional, for GPU nodes)
#   4. OSMO platform via Helm quick-start chart
#
# Usage:
#   bash scripts/03-deploy-osmo.sh [--config path/to/inventory.env] [--values path/to/osmo-values.yaml]

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

main() {
  local config_file=""
  local values_file=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        config_file="$2"
        shift 2
        ;;
      --values)
        values_file="$2"
        shift 2
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

  if [[ -z "${values_file}" ]]; then
    values_file="$(get_project_dir)/config/osmo-values.yaml"
  fi

  # Ensure KUBECONFIG is set
  export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"

  log_step "3" "Deploying NVIDIA OSMO on the cluster"

  check_command "kubectl"
  check_command "helm"

  verify_cluster
  apply_node_labels
  install_storage_provisioner
  install_kai_scheduler
  install_gpu_operator
  install_osmo "${values_file}"
  configure_access
  verify_deployment

  log_success "NVIDIA OSMO deployed successfully!"
  print_access_info
}

usage() {
  cat <<EOF
Usage: ${0##*/} [OPTIONS]

Options:
  --config PATH    Path to inventory.env configuration file
  --values PATH    Path to OSMO Helm values file (default: config/osmo-values.yaml)
  --help, -h       Show this help message
EOF
  exit 1
}

##############################################################################
# Cluster Verification
##############################################################################

verify_cluster() {
  log_info "Verifying Kubernetes cluster..."

  if ! kubectl cluster-info &>/dev/null; then
    log_error "Cannot connect to the Kubernetes cluster."
    log_info "Ensure KUBECONFIG is set and the cluster is running."
    exit 1
  fi

  local node_count
  node_count="$(kubectl get nodes --no-headers 2>/dev/null | wc -l)"
  log_success "Cluster is reachable with ${node_count} node(s)."

  kubectl get nodes
  echo ""
}

##############################################################################
# Node Labels
##############################################################################

apply_node_labels() {
  log_info "Applying node labels for OSMO scheduling..."

  local workers=("${WORKER_HOSTNAMES[@]}")
  local labels=("${WORKER_LABELS[@]}")

  for i in "${!workers[@]}"; do
    # kubelet always registers node names in lowercase, so normalize the
    # hostname before touching kubectl. Without this, inventories with
    # mixed-case hostnames (e.g. "MS-7D52-1") fail with
    # `Error from server (NotFound): nodes "MS-7D52-1" not found`.
    local node
    node="$(printf '%s' "${workers[$i]}" | tr '[:upper:]' '[:lower:]')"
    local label="${labels[$i]:-node_group=compute}"

    if kubectl get node "${node}" &>/dev/null; then
      kubectl label node "${node}" "${label}" --overwrite
      log_success "  ${node}: ${label}"
    else
      log_warning "  Node '${node}' not found in cluster. Skipping."
    fi
  done

  # Apply pending labels from worker join scripts
  local pending_labels="$(get_project_dir)/config/pending-labels.txt"
  if [[ -f "${pending_labels}" ]]; then
    log_info "Applying pending labels from worker joins..."
    while IFS= read -r cmd; do
      eval "${cmd}" 2>/dev/null || true
    done < "${pending_labels}"
    rm -f "${pending_labels}"
  fi

  # Label control plane for KAI scheduler if no dedicated node
  local has_kai_node=false
  for label in "${labels[@]}"; do
    if [[ "${label}" == *"kai-scheduler"* ]]; then
      has_kai_node=true
      break
    fi
  done

  if [[ "${has_kai_node}" == "false" ]]; then
    log_info "No dedicated KAI scheduler node. Using control plane."
    local cp_node
    cp_node="$(printf '%s' "${CONTROL_PLANE_HOSTNAME}" | tr '[:upper:]' '[:lower:]')"
    kubectl label node "${cp_node}" node_group=kai-scheduler --overwrite 2>/dev/null || true
  fi

  echo ""
}

##############################################################################
# Storage Provisioner (for bare-metal kubeadm clusters)
##############################################################################

install_storage_provisioner() {
  log_info "Checking for StorageClass provisioner..."

  if kubectl get storageclass 2>/dev/null | grep -q '(default)'; then
    log_info "Default StorageClass already exists. Skipping provisioner install."
    return
  fi

  log_info "Installing Rancher local-path-provisioner..."
  kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.30/deploy/local-path-storage.yaml

  # Set local-path as the default StorageClass
  kubectl patch storageclass local-path \
    -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

  log_success "local-path-provisioner installed and set as default StorageClass."
  echo ""
}

##############################################################################
# KAI Scheduler
##############################################################################

install_kai_scheduler() {
  log_info "Installing KAI Scheduler (${KAI_SCHEDULER_VERSION})..."

  # Check if already installed
  if helm list -n kai-scheduler --output json 2>/dev/null | jq -e '.[] | select(.name == "kai-scheduler")' &>/dev/null; then
    log_info "KAI Scheduler already installed. Skipping."
    return
  fi

  helm upgrade --install kai-scheduler \
    "oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler" \
    --version "${KAI_SCHEDULER_VERSION}" \
    --create-namespace -n kai-scheduler \
    --set "scheduler.additionalArgs[0]=--default-staleness-grace-period=-1s" \
    --set "scheduler.additionalArgs[1]=--update-pod-eviction-condition=true" \
    --wait --timeout 5m

  log_success "KAI Scheduler installed."
}

##############################################################################
# GPU Operator (Optional)
##############################################################################

install_gpu_operator() {
  if [[ "${ENABLE_GPU}" != "true" ]]; then
    log_info "GPU support not enabled. Skipping GPU Operator installation."
    return
  fi

  log_info "Installing NVIDIA GPU Operator (${GPU_OPERATOR_VERSION})..."

  helm repo add nvidia https://nvidia.github.io/gpu-operator 2>/dev/null || true
  helm repo update

  helm upgrade --install gpu-operator nvidia/gpu-operator \
    --namespace gpu-operator \
    --create-namespace \
    --set driver.enabled=true \
    --set toolkit.enabled=true \
    --set nfd.enabled=true \
    --wait --timeout 10m

  log_success "NVIDIA GPU Operator installed."
}

##############################################################################
# OSMO Installation
##############################################################################

install_osmo() {
  local values_file="$1"

  log_info "Installing NVIDIA OSMO..."

  # Clean up any stale resources from previous installs
  cleanup_stale_resources

  # Add OSMO Helm repository
  helm repo add osmo https://helm.ngc.nvidia.com/nvidia/osmo 2>/dev/null || true
  helm repo update

  # Build Helm command (--wait=false so we can monitor with early failure checks)
  local helm_cmd=(
    "helm" "upgrade" "--install" "osmo" "osmo/quick-start"
    "--namespace" "${OSMO_NAMESPACE}"
    "--create-namespace"
    "--wait=false"
  )

  # Apply custom values if the file exists
  if [[ -f "${values_file}" ]]; then
    log_info "Using custom values from: ${values_file}"
    helm_cmd+=("-f" "${values_file}")
  else
    log_info "No custom values file found. Using chart defaults."
  fi

  # Override nodeSelectors via --set-json (the chart ignores nodeSelector: {} in values file)
  log_info "Clearing hardcoded nodeSelector constraints for bare-metal deployment..."
  helm_cmd+=(
    "--set-json" "global.nodeSelector={}"
    "--set-json" "ingress-nginx.controller.nodeSelector={\"kubernetes.io/os\":\"linux\"}"
    "--set-json" "service.services.postgres.nodeSelector={}"
    "--set-json" "service.services.redis.nodeSelector={}"
    "--set-json" "service.services.localstackS3.nodeSelector={}"
  )

  "${helm_cmd[@]}"
  log_success "Helm release installed."

  # Remove any remaining nodeSelector constraints from all deployments
  patch_node_selectors

  # Monitor deployment with early failure detection
  monitor_deployment 600
}

##############################################################################
# Cleanup stale resources from previous installs
##############################################################################

cleanup_stale_resources() {
  log_info "Checking for stale resources from previous installs..."

  # Remove a lingering standalone ingress-nginx release. Older versions of
  # 01-init-control-plane.sh installed one at NodePort 30080; OSMO's
  # quick-start chart now owns that port, so the two conflict with:
  #   'Service "quick-start" is invalid: spec.ports[0].nodePort:
  #    Invalid value: 30080: provided port is already allocated'
  if helm status -n ingress-nginx ingress-nginx &>/dev/null; then
    log_warning "Removing legacy 'ingress-nginx' Helm release (conflicts with OSMO quick-start on :30080)..."
    helm uninstall -n ingress-nginx ingress-nginx 2>/dev/null || true
    kubectl delete namespace ingress-nginx --wait=false 2>/dev/null || true
  fi

  # Remove stale validating webhooks that block new installs
  local stale_webhooks
  stale_webhooks=$(kubectl get validatingwebhookconfigurations -o name 2>/dev/null \
    | grep -E 'ingress-nginx|quick-start|osmo' || true)
  if [[ -n "${stale_webhooks}" ]]; then
    log_warning "Removing stale webhook configurations..."
    echo "${stale_webhooks}" | xargs kubectl delete 2>/dev/null || true
  fi

  # Remove stale IngressClass
  if kubectl get ingressclass nginx &>/dev/null; then
    log_warning "Removing stale IngressClass 'nginx'..."
    kubectl delete ingressclass nginx 2>/dev/null || true
  fi

  # Delete any completed/failed Jobs from previous installs. Kubernetes Job
  # pod templates are immutable, so a Helm `upgrade` that changes a Job spec
  # (e.g. different osmoImageTag or env) fails with:
  #   `Job.batch "osmo-config-setup" is invalid: spec.template: field is immutable`
  # Removing the stale Job first lets Helm recreate it cleanly.
  if kubectl get namespace "${OSMO_NAMESPACE}" &>/dev/null; then
    local stale_jobs
    stale_jobs="$(kubectl get jobs -n "${OSMO_NAMESPACE}" \
                    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
    if [[ -n "${stale_jobs}" ]]; then
      log_warning "Removing stale Jobs (pod templates are immutable): ${stale_jobs}"
      # shellcheck disable=SC2086
      kubectl delete job -n "${OSMO_NAMESPACE}" ${stale_jobs} --ignore-not-found 2>/dev/null || true
    fi
  fi

  # Wait briefly for the NodePort allocator to release :30080 after the
  # namespace/services above are torn down. Without this, the very next
  # `helm install osmo` can still see the port as allocated.
  local tries=0
  while [[ ${tries} -lt 15 ]]; do
    if ! kubectl get svc -A -o jsonpath='{range .items[*]}{.spec.ports[*].nodePort}{"\n"}{end}' 2>/dev/null \
         | grep -qx '30080'; then
      break
    fi
    sleep 2
    ((tries++))
  done

  log_info "Stale resource cleanup complete."
}

##############################################################################
# Patch node selectors off all deployments
##############################################################################

patch_node_selectors() {
  log_info "Removing hardcoded nodeSelector from deployments..."

  local patched=0
  local skipped=0

  # The chart also applies nodeSelectors to StatefulSets and Jobs (e.g. redis,
  # postgres, config-setup). Strip them off every workload type so nothing
  # gets stuck in Pending on bare-metal deployments that don't use the
  # upstream node_group= taxonomy.
  local kinds=(deployment statefulset daemonset job)
  for kind in "${kinds[@]}"; do
    # `|| true` guards against `set -e` exits when a kind has no objects.
    local names
    names="$(kubectl get "${kind}" -n "${OSMO_NAMESPACE}" \
             -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)"
    for obj in ${names}; do
      local ns
      ns="$(kubectl get "${kind}" "${obj}" -n "${OSMO_NAMESPACE}" \
            -o jsonpath='{.spec.template.spec.nodeSelector}' 2>/dev/null || true)"
      if [[ -n "${ns}" && "${ns}" != "{}" ]]; then
        # NOTE: use `patched=$((patched+1))` — `((patched++))` evaluates to 0
        # on the first iteration and trips `set -e`, silently aborting the
        # rest of the loop and leaving pods stuck in Pending.
        if kubectl patch "${kind}" "${obj}" -n "${OSMO_NAMESPACE}" --type=json \
          -p='[{"op": "remove", "path": "/spec/template/spec/nodeSelector"}]' &>/dev/null; then
          patched=$((patched + 1))
        fi
      else
        skipped=$((skipped + 1))
      fi
    done
  done

  log_success "Patched ${patched} workload(s), ${skipped} already clean."
}

##############################################################################
# Deployment Monitoring with Early Failure Detection
##############################################################################

monitor_deployment() {
  local timeout="${1:-600}"
  local interval=10
  local elapsed=0
  local last_status=""

  log_info "Monitoring OSMO deployment (timeout: ${timeout}s)..."
  echo ""

  while (( elapsed < timeout )); do
    local total=0 ready=0 pending=0 failed=0 creating=0 init=0
    local scheduling_failures="" image_failures="" crash_failures=""

    # Count pod states
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      ((total++))
      local name status
      name=$(echo "${line}" | awk '{print $1}')
      status=$(echo "${line}" | awk '{print $3}')

      case "${status}" in
        Running)
          local ready_col
          ready_col=$(echo "${line}" | awk '{print $2}')
          local ready_count total_count
          ready_count="${ready_col%%/*}"
          total_count="${ready_col##*/}"
          if [[ "${ready_count}" == "${total_count}" ]]; then
            ((ready++))
          else
            ((creating++))
          fi
          ;;
        Pending)
          ((pending++))
          # Check for scheduling failures
          local events
          events=$(kubectl describe pod "${name}" -n "${OSMO_NAMESPACE}" 2>/dev/null \
            | grep -A1 "FailedScheduling" || true)
          if [[ -n "${events}" ]]; then
            scheduling_failures+="  ${name}\n"
          fi
          ;;
        Init:*|PodInitializing)
          ((init++))
          ;;
        ImagePullBackOff|ErrImagePull)
          ((failed++))
          image_failures+="  ${name}: ${status}\n"
          ;;
        CrashLoopBackOff|Error)
          ((failed++))
          crash_failures+="  ${name}: ${status}\n"
          ;;
        Completed|Succeeded)
          ((ready++))
          ;;
        *)
          ((creating++))
          ;;
      esac
    done < <(kubectl get pods -n "${OSMO_NAMESPACE}" --no-headers 2>/dev/null)

    # Print concise status line (overwrite previous)
    local status_msg
    status_msg=$(printf "[%3ds/%ds] Pods: %d/%d ready | %d pending | %d init | %d creating | %d failed" \
      "${elapsed}" "${timeout}" "${ready}" "${total}" "${pending}" "${init}" "${creating}" "${failed}")
    printf "\r%-100s" "${status_msg}"

    # === EARLY FAILURE CHECKS ===

    # Check 1: Scheduling failures after 30s (nodeSelector/affinity mismatch)
    if (( elapsed >= 30 && pending > 0 )); then
      if [[ -n "${scheduling_failures}" ]]; then
        echo ""
        log_error "SCHEDULING FAILURE detected! Pods cannot be placed on any node:"
        echo -e "${scheduling_failures}"
        log_info "Common causes:"
        log_info "  - nodeSelector labels don't match any node"
        log_info "  - Node affinity rules exclude all nodes"
        log_info "  - Insufficient resources (CPU/memory)"
        log_info ""
        log_info "Current node labels:"
        kubectl get nodes --show-labels 2>/dev/null | awk '{print "  " $0}'
        echo ""
        log_error "Fix the scheduling issue and re-run deployment."
        exit 1
      fi
    fi

    # Check 2: Image pull failures (bad image ref or registry auth)
    if [[ -n "${image_failures}" ]]; then
      echo ""
      log_error "IMAGE PULL FAILURE detected! Cannot download container images:"
      echo -e "${image_failures}"
      log_info "Common causes:"
      log_info "  - Incorrect image name/tag"
      log_info "  - Missing image pull secret (registry auth)"
      log_info "  - No internet access from nodes"
      log_error "Fix registry access and re-run deployment."
      exit 1
    fi

    # Check 3: CrashLoopBackOff (application errors)
    if (( elapsed >= 60 )) && [[ -n "${crash_failures}" ]]; then
      echo ""
      log_error "CRASH LOOP detected! Pods are crashing repeatedly:"
      echo -e "${crash_failures}"
      log_info "Check pod logs:"
      echo -e "${crash_failures}" | while read -r crash_line; do
        local crash_pod
        crash_pod=$(echo "${crash_line}" | awk -F: '{print $1}' | xargs)
        [[ -n "${crash_pod}" ]] && log_info "  kubectl logs ${crash_pod} -n ${OSMO_NAMESPACE}"
      done
      log_error "Fix application errors and re-run deployment."
      exit 1
    fi

    # Check 4: PVC binding issues after 45s
    if (( elapsed >= 45 )); then
      local unbound_pvcs
      unbound_pvcs=$(kubectl get pvc -n "${OSMO_NAMESPACE}" --no-headers 2>/dev/null \
        | grep -v "Bound" | awk '{print "  " $1 ": " $2}' || true)
      if [[ -n "${unbound_pvcs}" ]]; then
        echo ""
        log_error "PVC BINDING FAILURE! Persistent volumes are not binding:"
        echo "${unbound_pvcs}"
        log_info "Common causes:"
        log_info "  - No default StorageClass configured"
        log_info "  - StorageClass provisioner not running"
        log_info "Current StorageClasses:"
        kubectl get storageclass 2>/dev/null | sed 's/^/  /'
        log_error "Fix storage and re-run deployment."
        exit 1
      fi
    fi

    # Check 5: All pods ready - success!
    if (( total > 0 && ready == total )); then
      echo ""
      log_success "All ${total} pods are ready!"
      return 0
    fi

    sleep "${interval}"
    ((elapsed += interval))
  done

  # Timeout
  echo ""
  log_error "TIMEOUT: Deployment did not complete within ${timeout}s."
  echo ""
  log_info "Final pod status:"
  kubectl get pods -n "${OSMO_NAMESPACE}" 2>/dev/null
  echo ""
  log_info "Pods not ready:"
  kubectl get pods -n "${OSMO_NAMESPACE}" --no-headers 2>/dev/null \
    | grep -v "Running\|Completed\|Succeeded" | sed 's/^/  /' || true
  exit 1
}

##############################################################################
# Access Configuration
##############################################################################

configure_access() {
  log_info "Configuring host entry for OSMO access..."

  local hosts_entry="127.0.0.1 ${OSMO_HOSTNAME}"

  if grep -q "${OSMO_HOSTNAME}" /etc/hosts 2>/dev/null; then
    log_info "Host entry already exists in /etc/hosts."
  else
    if [[ "$(id -u)" -eq 0 ]]; then
      echo "${hosts_entry}" >> /etc/hosts
      log_success "Added '${hosts_entry}' to /etc/hosts."
    else
      log_warning "Add the following line to /etc/hosts manually (requires sudo):"
      log_info "  ${hosts_entry}"
    fi
  fi
}

##############################################################################
# Deployment Verification
##############################################################################

verify_deployment() {
  log_info "Verifying OSMO deployment..."

  echo ""
  log_info "Pods in namespace '${OSMO_NAMESPACE}':"
  kubectl get pods --namespace "${OSMO_NAMESPACE}" 2>/dev/null || true

  echo ""
  log_info "Services in namespace '${OSMO_NAMESPACE}':"
  kubectl get services --namespace "${OSMO_NAMESPACE}" 2>/dev/null || true
  echo ""
}

##############################################################################
# Access Information
##############################################################################

print_access_info() {
  local control_ip="${CONTROL_PLANE_IP}"

  cat <<EOF

==============================================================================
                    NVIDIA OSMO Deployment Complete!
==============================================================================

  OSMO Web UI:
    From control plane: http://${OSMO_HOSTNAME}
    Via port-forward:   kubectl port-forward svc/osmo-ui 3000:80 -n ${OSMO_NAMESPACE}
                        Then visit http://localhost:3000

  OSMO API:
    Via port-forward:   kubectl port-forward svc/osmo-service 9000:80 -n ${OSMO_NAMESPACE}
                        Then visit http://localhost:9000/api/docs

  Install the OSMO CLI:
    bash scripts/04-install-cli.sh

  Login:
    osmo login http://${OSMO_HOSTNAME} --method=dev --username=testuser

  Cluster Nodes:
$(kubectl get nodes 2>/dev/null | sed 's/^/    /')

==============================================================================
EOF
}

main "$@"
