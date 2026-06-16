#!/usr/bin/env bash
#
# 06-deploy-local.sh
# All-in-one OSMO deployment on local machines (1 or 2+ nodes).
#
# This script automates the entire deployment lifecycle:
#   1. Install prerequisites (containerd, kubeadm, kubelet, kubectl)
#   2. Initialize a Kubernetes cluster on this machine (control plane)
#   3. Untaint control plane to run OSMO services and data
#   4. (If workers configured) SSH to worker nodes, install prereqs, join them
#   5. Install storage provisioner, KAI Scheduler, and (optionally) GPU Operator
#   6. Deploy OSMO via Helm quick-start chart
#   7. Install the OSMO CLI
#
# Usage:
#   sudo bash scripts/06-deploy-local.sh [OPTIONS]
#
# Options:
#   --config PATH       Path to inventory.env (default: config/inventory.local.env)
#   --values PATH       Path to osmo-values.yaml (default: config/osmo-values.yaml)
#   --skip-prereqs      Skip prerequisite installation (use if already installed)
#   --skip-k8s          Skip Kubernetes init (use if cluster already running)
#   --gpu               Enable GPU Operator installation
#   --help, -h          Show this help message

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

main() {
  local config_file=""
  local values_file=""
  local skip_prereqs=false
  local skip_k8s=false
  local enable_gpu=""

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
      --skip-prereqs)
        skip_prereqs=true
        shift
        ;;
      --skip-k8s)
        skip_k8s=true
        shift
        ;;
      --gpu)
        enable_gpu="true"
        shift
        ;;
      --help|-h)
        usage
        ;;
      *)
        log_warning "Unknown option: $1"
        shift
        ;;
    esac
  done

  # Default to local config
  if [[ -z "${config_file}" ]]; then
    config_file="$(get_project_dir)/config/inventory.env"
    if [[ ! -f "${config_file}" ]]; then
      config_file="$(get_project_dir)/config/inventory.local.env"
    fi
  fi

  load_config "${config_file}"

  # Override GPU setting if flag passed
  if [[ -n "${enable_gpu}" ]]; then
    ENABLE_GPU="${enable_gpu}"
  fi

  if [[ -z "${values_file}" ]]; then
    values_file="$(get_project_dir)/config/osmo-values.yaml"
  fi

  # Detect the machine's primary IP if configured as 127.0.0.1
  local advertise_ip="${CONTROL_PLANE_IP}"
  if [[ "${advertise_ip}" == "127.0.0.1" ]]; then
    advertise_ip="$(detect_local_ip)"
    log_info "Using detected local IP for API server: ${advertise_ip}"
  fi

  log_step "0" "OSMO Local Deployment"
  print_system_info

  # Require root for prerequisites and k8s init
  if [[ "${skip_prereqs}" == "false" || "${skip_k8s}" == "false" ]]; then
    check_root
  fi

  #--- Phase 1: Prerequisites ---
  if [[ "${skip_prereqs}" == "false" ]]; then
    log_step "1" "Installing prerequisites"
    bash "${SCRIPT_DIR}/00-prerequisites.sh"
  else
    log_info "Skipping prerequisites (--skip-prereqs)."
  fi

  #--- Phase 2: Kubernetes cluster ---
  if [[ "${skip_k8s}" == "false" ]]; then
    log_step "2" "Initializing Kubernetes cluster (control plane)"
    init_single_node_cluster "${advertise_ip}"
  else
    log_info "Skipping Kubernetes init (--skip-k8s)."
    # Ensure KUBECONFIG is set
    export KUBECONFIG="${KUBECONFIG:-/etc/kubernetes/admin.conf}"
  fi

  #--- Phase 2b: Join worker nodes (if configured) ---
  join_configured_workers

  #--- Phase 3: Deploy OSMO stack ---
  log_step "3" "Deploying OSMO stack"
  ensure_helm
  install_storage_provisioner
  install_kai_scheduler
  install_gpu_operator_if_needed
  install_osmo "${values_file}"

  #--- Phase 4: Post-deployment ---
  log_step "4" "Post-deployment setup"
  configure_hosts
  install_osmo_cli

  #--- Phase 5: Verify ---
  log_step "5" "Verifying deployment"
  verify_deployment

  print_complete_message
}

usage() {
  cat <<EOF
Usage: sudo ${0##*/} [OPTIONS]

Deploy NVIDIA OSMO on local machines (1 control plane + optional workers).

Options:
  --config PATH       Inventory config file (default: config/inventory.local.env)
  --values PATH       OSMO Helm values (default: config/osmo-values.yaml)
  --skip-prereqs      Skip prerequisite installation
  --skip-k8s          Skip Kubernetes cluster initialization
  --gpu               Enable NVIDIA GPU Operator
  --help, -h          Show this help

Examples:
  # Full deployment (control plane + workers from config)
  sudo bash scripts/06-deploy-local.sh

  # Full deployment with GPU support
  sudo bash scripts/06-deploy-local.sh --gpu

  # Redeploy OSMO only (cluster already exists)
  sudo bash scripts/06-deploy-local.sh --skip-prereqs --skip-k8s

  # Use custom config
  sudo bash scripts/06-deploy-local.sh --config /path/to/inventory.env
EOF
  exit 0
}

##############################################################################
# System Detection
##############################################################################

detect_local_ip() {
  # Get the default route interface IP
  local ip=""
  ip="$(ip -4 route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+' || true)"

  if [[ -z "${ip}" ]]; then
    # Fallback: first non-loopback IPv4 address
    ip="$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")"
  fi

  echo "${ip}"
}

print_system_info() {
  log_info "System information:"
  echo "  Hostname:  $(hostname)"
  echo "  OS:        $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME}" || uname -s)"
  echo "  Kernel:    $(uname -r)"
  echo "  CPU:       $(nproc) cores"
  echo "  Memory:    $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo 'unknown')"
  echo "  Disk:      $(df -h / 2>/dev/null | awk 'NR==2{print $4 " available"}' || echo 'unknown')"

  # GPU detection
  if command -v nvidia-smi &>/dev/null; then
    echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'detected')"
  else
    echo "  GPU:       not detected"
  fi
  echo ""

  # Minimum requirements check
  local cores
  cores="$(nproc)"
  local mem_kb
  mem_kb="$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)"
  local mem_gb=$((mem_kb / 1024 / 1024))

  if [[ "${cores}" -lt 4 ]]; then
    log_warning "Minimum 4 CPU cores recommended. Detected: ${cores}"
  fi
  if [[ "${mem_gb}" -lt 8 ]]; then
    log_warning "Minimum 8 GB RAM recommended. Detected: ${mem_gb} GB"
  fi
}

##############################################################################
# Single-Node Kubernetes
##############################################################################

init_single_node_cluster() {
  local advertise_ip="$1"

  if [[ -f /etc/kubernetes/admin.conf ]]; then
    log_warning "Kubernetes cluster already initialized."
    log_info "To reinitialize, run: sudo kubeadm reset -f"
    export KUBECONFIG=/etc/kubernetes/admin.conf
    configure_kubectl_for_user
    if [[ "${UNTAINT_CONTROL_PLANE:-true}" == "true" || "${SINGLE_NODE}" == "true" ]]; then
      untaint_control_plane
    fi
    return
  fi

  log_info "Initializing Kubernetes cluster (control plane)..."
  log_info "  API server address: ${advertise_ip}"
  log_info "  Pod CIDR: ${POD_NETWORK_CIDR}"
  log_info "  Service CIDR: ${SERVICE_CIDR}"

  kubeadm init \
    --apiserver-advertise-address="${advertise_ip}" \
    --pod-network-cidr="${POD_NETWORK_CIDR}" \
    --service-cidr="${SERVICE_CIDR}" \
    --node-name="${CONTROL_PLANE_HOSTNAME}" \
    --cri-socket="unix:///run/containerd/containerd.sock"

  export KUBECONFIG=/etc/kubernetes/admin.conf

  log_success "Kubernetes cluster initialized."

  configure_kubectl_for_user
  install_cni
  if [[ "${UNTAINT_CONTROL_PLANE:-true}" == "true" || "${SINGLE_NODE}" == "true" ]]; then
    untaint_control_plane
  fi
  wait_for_node_ready
}

configure_kubectl_for_user() {
  log_info "Configuring kubectl access..."

  local user_home
  user_home="$(eval echo "~${SUDO_USER:-root}")"
  local kube_dir="${user_home}/.kube"

  mkdir -p "${kube_dir}"
  cp -f /etc/kubernetes/admin.conf "${kube_dir}/config"

  if [[ -n "${SUDO_USER:-}" ]]; then
    chown -R "${SUDO_USER}:$(id -gn "${SUDO_USER}")" "${kube_dir}"
  fi

  # Also save to project config
  local project_dir
  project_dir="$(get_project_dir)"
  cp -f /etc/kubernetes/admin.conf "${project_dir}/config/kubeconfig"
  chmod 0600 "${project_dir}/config/kubeconfig"

  if [[ -n "${SUDO_USER:-}" ]]; then
    chown "${SUDO_USER}:$(id -gn "${SUDO_USER}")" "${project_dir}/config/kubeconfig"
  fi

  log_success "kubectl configured for user '${SUDO_USER:-root}'."
}

install_cni() {
  log_info "Installing Calico CNI plugin..."

  kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/calico.yaml

  log_info "Waiting for Calico pods to start..."
  sleep 15
  kubectl wait --for=condition=Ready pods --all \
    --namespace kube-system --timeout=300s 2>/dev/null || true

  log_success "Calico CNI installed."
}

untaint_control_plane() {
  log_info "Removing scheduling restrictions from control plane node..."

  # Remove the NoSchedule taint so workloads can run on this node
  kubectl taint nodes "${CONTROL_PLANE_HOSTNAME}" \
    node-role.kubernetes.io/control-plane:NoSchedule- 2>/dev/null || true

  # Also remove the older taint key (pre-1.24)
  kubectl taint nodes "${CONTROL_PLANE_HOSTNAME}" \
    node-role.kubernetes.io/master:NoSchedule- 2>/dev/null || true

  # Label the node for OSMO services and data (runs on control plane)
  kubectl label node "${CONTROL_PLANE_HOSTNAME}" \
    node_group=service --overwrite 2>/dev/null || true
  kubectl label node "${CONTROL_PLANE_HOSTNAME}" \
    node_group=data --overwrite 2>/dev/null || true

  log_success "Control plane node can now schedule workloads."
}

wait_for_node_ready() {
  log_info "Waiting for node to become Ready..."

  kubectl wait --for=condition=Ready nodes --all --timeout=120s 2>/dev/null || true

  kubectl get nodes -o wide
  echo ""
}

##############################################################################
# Worker Node Joining
##############################################################################

join_configured_workers() {
  # Check if there are workers to join
  if [[ -z "${WORKER_IPS+x}" ]] || [[ ${#WORKER_IPS[@]} -eq 0 ]]; then
    log_info "No worker nodes configured. Running as single-node cluster."
    # Ensure control plane also has the compute label for single-node
    kubectl label node "${CONTROL_PLANE_HOSTNAME}" \
      node_group=compute --overwrite 2>/dev/null || true
    return
  fi

  local worker_count=${#WORKER_IPS[@]}
  log_step "2b" "Joining ${worker_count} worker node(s) to the cluster"

  # Get SSH key path
  local ssh_key="${SSH_KEY_PATH:-${HOME}/.ssh/id_rsa}"
  # Expand ~ if present
  ssh_key="${ssh_key/#\~/$HOME}"

  # Generate join command
  log_info "Generating join token..."
  local join_cmd
  join_cmd="$(kubeadm token create --print-join-command)"

  # Save join command for reference
  local project_dir
  project_dir="$(get_project_dir)"
  echo "${join_cmd}" > "${project_dir}/config/join-command.sh"
  chmod 0600 "${project_dir}/config/join-command.sh"

  for i in "${!WORKER_IPS[@]}"; do
    local w_ip="${WORKER_IPS[$i]}"
    local w_user="${WORKER_USERS[$i]:-ubuntu}"
    local w_hostname="${WORKER_HOSTNAMES[$i]:-worker-$((i + 1))}"
    local w_labels="${WORKER_LABELS[$i]:-node_group=compute}"

    log_info "--- Worker $((i + 1)): ${w_ip} (${w_hostname}) ---"

    # Check SSH connectivity
    if ! ssh -i "${ssh_key}" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
      "${w_user}@${w_ip}" "echo 'SSH OK'" &>/dev/null; then
      log_error "Cannot SSH to ${w_user}@${w_ip}. Skipping this node."
      log_info "Ensure SSH key is authorized: ssh-copy-id -i ${ssh_key} ${w_user}@${w_ip}"
      continue
    fi

    # Copy scripts to the worker
    log_info "Copying scripts to ${w_ip}..."
    ssh -i "${ssh_key}" -o StrictHostKeyChecking=no \
      "${w_user}@${w_ip}" "mkdir -p ~/osmo-deployment/scripts ~/osmo-deployment/config" 2>/dev/null

    scp -i "${ssh_key}" -o StrictHostKeyChecking=no -r \
      "${SCRIPT_DIR}/." "${w_user}@${w_ip}:~/osmo-deployment/scripts/" 2>/dev/null

    ssh -i "${ssh_key}" -o StrictHostKeyChecking=no \
      "${w_user}@${w_ip}" "chmod +x ~/osmo-deployment/scripts/*.sh" 2>/dev/null

    # Install prerequisites
    log_info "Installing prerequisites on ${w_ip}..."
    ssh -i "${ssh_key}" -o StrictHostKeyChecking=no \
      "${w_user}@${w_ip}" "sudo bash ~/osmo-deployment/scripts/00-prerequisites.sh" 2>&1

    # Join cluster
    log_info "Joining ${w_ip} to the cluster..."
    ssh -i "${ssh_key}" -o StrictHostKeyChecking=no \
      "${w_user}@${w_ip}" "sudo ${join_cmd} --cri-socket='unix:///run/containerd/containerd.sock'" 2>&1 || {
      log_warning "Join may have failed for ${w_ip}. Check connectivity and try again."
      continue
    }

    # Wait for the node to register
    log_info "Waiting for ${w_hostname} to register..."
    local attempts=0
    while [[ ${attempts} -lt 30 ]]; do
      if kubectl get node "${w_hostname}" &>/dev/null 2>&1; then
        break
      fi
      attempts=$((attempts + 1))
      sleep 5
    done

    # Apply labels
    log_info "Applying label '${w_labels}' to ${w_hostname}..."
    kubectl label node "${w_hostname}" "${w_labels}" --overwrite 2>/dev/null || true

    log_success "Worker ${w_hostname} (${w_ip}) joined."
    echo ""
  done

  # Wait for all nodes to be ready
  log_info "Waiting for all nodes to become Ready..."
  kubectl wait --for=condition=Ready nodes --all --timeout=300s 2>/dev/null || true
  kubectl get nodes -o wide
  echo ""
}

##############################################################################
# Helm
##############################################################################

ensure_helm() {
  if command -v helm &>/dev/null; then
    log_info "Helm already installed: $(helm version --short 2>/dev/null || echo 'available')"
    return
  fi

  log_info "Installing Helm..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

  log_success "Helm installed."
}

##############################################################################
# Storage Provisioner
##############################################################################

install_storage_provisioner() {
  log_info "Setting up storage provisioner..."

  if kubectl get storageclass 2>/dev/null | grep -q '(default)'; then
    log_info "Default StorageClass already exists."
    return
  fi

  kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.30/deploy/local-path-storage.yaml

  kubectl patch storageclass local-path \
    -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

  log_success "local-path-provisioner installed as default StorageClass."
}

##############################################################################
# KAI Scheduler
##############################################################################

install_kai_scheduler() {
  log_info "Installing KAI Scheduler (${KAI_SCHEDULER_VERSION})..."

  if helm list -n kai-scheduler --output json 2>/dev/null | jq -e '.[] | select(.name == "kai-scheduler")' &>/dev/null; then
    log_info "KAI Scheduler already installed. Skipping."
    return
  fi

  # Label control plane for KAI scheduler
  kubectl label node "${CONTROL_PLANE_HOSTNAME}" \
    node_group=kai-scheduler --overwrite 2>/dev/null || true

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
# GPU Operator
##############################################################################

install_gpu_operator_if_needed() {
  if [[ "${ENABLE_GPU}" != "true" ]]; then
    log_info "GPU Operator not enabled. Skipping."
    if command -v nvidia-smi &>/dev/null; then
      log_info "Hint: NVIDIA GPU detected. Use --gpu flag to enable GPU Operator."
    fi
    return
  fi

  log_info "Installing NVIDIA GPU Operator (${GPU_OPERATOR_VERSION})..."

  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia 2>/dev/null || true
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
# OSMO
##############################################################################

install_osmo() {
  local values_file="$1"

  log_info "Installing NVIDIA OSMO (quick-start chart)..."

  # Clean up stale resources from previous installs
  log_info "Checking for stale webhooks/ingress classes..."
  kubectl get validatingwebhookconfigurations -o name 2>/dev/null \
    | grep -E 'ingress-nginx|quick-start|osmo' \
    | xargs kubectl delete 2>/dev/null || true
  kubectl delete ingressclass nginx 2>/dev/null || true

  helm repo add osmo https://helm.ngc.nvidia.com/nvidia/osmo 2>/dev/null || true
  helm repo update

  local helm_cmd=(
    "helm" "upgrade" "--install" "osmo" "osmo/quick-start"
    "--namespace" "${OSMO_NAMESPACE}"
    "--create-namespace"
    "--wait=false"
  )

  if [[ -f "${values_file}" ]]; then
    log_info "Using custom values from: ${values_file}"
    helm_cmd+=("-f" "${values_file}")
  fi

  # Set image tag if specified
  if [[ -n "${OSMO_IMAGE_TAG:-}" && "${OSMO_IMAGE_TAG}" != "latest" ]]; then
    helm_cmd+=("--set-string" "global.osmoImageTag=${OSMO_IMAGE_TAG}")
  fi

  # Override nodeSelectors via --set-json (chart ignores nodeSelector: {} in values file)
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
  log_info "Removing hardcoded nodeSelector from deployments..."
  local patched=0
  for dep in $(kubectl get deployments -n "${OSMO_NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    local ns
    ns=$(kubectl get deployment "${dep}" -n "${OSMO_NAMESPACE}" \
      -o jsonpath='{.spec.template.spec.nodeSelector}' 2>/dev/null)
    if [[ -n "${ns}" && "${ns}" != "{}" ]]; then
      kubectl patch deployment "${dep}" -n "${OSMO_NAMESPACE}" --type=json \
        -p='[{"op": "remove", "path": "/spec/template/spec/nodeSelector"}]' &>/dev/null && ((patched++))
    fi
  done
  log_success "Patched ${patched} deployment(s)."

  # Monitor deployment with early failure detection
  monitor_osmo_deployment 600

  log_success "NVIDIA OSMO installed."
}

##############################################################################
# OSMO Deployment Monitoring with Early Failure Detection
##############################################################################

monitor_osmo_deployment() {
  local timeout="${1:-600}"
  local interval=10
  local elapsed=0

  log_info "Monitoring OSMO deployment (timeout: ${timeout}s)..."
  echo ""

  while (( elapsed < timeout )); do
    local total=0 ready=0 pending=0 failed=0 creating=0 init=0
    local scheduling_failures="" image_failures="" crash_failures=""

    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      ((total++))
      local name status
      name=$(echo "${line}" | awk '{print $1}')
      status=$(echo "${line}" | awk '{print $3}')

      case "${status}" in
        Running)
          local rc
          rc=$(echo "${line}" | awk '{print $2}')
          if [[ "${rc%%/*}" == "${rc##*/}" ]]; then ((ready++)); else ((creating++)); fi
          ;;
        Pending)
          ((pending++))
          if kubectl describe pod "${name}" -n "${OSMO_NAMESPACE}" 2>/dev/null | grep -q "FailedScheduling"; then
            scheduling_failures+="  ${name}\n"
          fi
          ;;
        Init:*|PodInitializing) ((init++)) ;;
        ImagePullBackOff|ErrImagePull) ((failed++)); image_failures+="  ${name}: ${status}\n" ;;
        CrashLoopBackOff|Error) ((failed++)); crash_failures+="  ${name}: ${status}\n" ;;
        Completed|Succeeded) ((ready++)) ;;
        *) ((creating++)) ;;
      esac
    done < <(kubectl get pods -n "${OSMO_NAMESPACE}" --no-headers 2>/dev/null)

    printf "\r[%3ds/%ds] Pods: %d/%d ready | %d pending | %d init | %d failed   " \
      "${elapsed}" "${timeout}" "${ready}" "${total}" "${pending}" "${init}" "${failed}"

    # Scheduling failure after 30s
    if (( elapsed >= 30 && pending > 0 )) && [[ -n "${scheduling_failures}" ]]; then
      echo ""; log_error "SCHEDULING FAILURE! Pods cannot be placed on any node:"
      echo -e "${scheduling_failures}"
      log_info "Check node labels: kubectl get nodes --show-labels"
      exit 1
    fi

    # Image pull failure
    if [[ -n "${image_failures}" ]]; then
      echo ""; log_error "IMAGE PULL FAILURE:"
      echo -e "${image_failures}"; exit 1
    fi

    # Crash loop after 60s
    if (( elapsed >= 60 )) && [[ -n "${crash_failures}" ]]; then
      echo ""; log_error "CRASH LOOP detected:"
      echo -e "${crash_failures}"; exit 1
    fi

    # PVC binding after 45s
    if (( elapsed >= 45 )); then
      local unbound
      unbound=$(kubectl get pvc -n "${OSMO_NAMESPACE}" --no-headers 2>/dev/null | grep -v "Bound" || true)
      if [[ -n "${unbound}" ]]; then
        echo ""; log_error "PVC BINDING FAILURE:"
        echo "${unbound}" | sed 's/^/  /'; exit 1
      fi
    fi

    # All ready
    if (( total > 0 && ready == total )); then
      echo ""; log_success "All ${total} pods are ready!"; return 0
    fi

    sleep "${interval}"
    ((elapsed += interval))
  done

  echo ""; log_error "TIMEOUT: Deployment did not complete within ${timeout}s."
  kubectl get pods -n "${OSMO_NAMESPACE}" 2>/dev/null
  exit 1
}

##############################################################################
# Host Configuration
##############################################################################

configure_hosts() {
  log_info "Configuring /etc/hosts for OSMO access..."

  local hosts_entry="127.0.0.1 ${OSMO_HOSTNAME}"

  if grep -q "${OSMO_HOSTNAME}" /etc/hosts 2>/dev/null; then
    log_info "Host entry '${OSMO_HOSTNAME}' already exists."
  else
    echo "${hosts_entry}" >> /etc/hosts
    log_success "Added '${hosts_entry}' to /etc/hosts."
  fi
}

##############################################################################
# OSMO CLI
##############################################################################

install_osmo_cli() {
  log_info "Installing OSMO CLI..."

  # Run cli install as the actual user, not root
  if [[ -n "${SUDO_USER:-}" ]]; then
    sudo -u "${SUDO_USER}" bash -c \
      'curl -fsSL https://raw.githubusercontent.com/NVIDIA/OSMO/refs/heads/main/install.sh | bash' || {
      log_warning "OSMO CLI installation failed. You can install manually later:"
      log_info "  curl -fsSL https://raw.githubusercontent.com/NVIDIA/OSMO/refs/heads/main/install.sh | bash"
      return
    }
  else
    curl -fsSL https://raw.githubusercontent.com/NVIDIA/OSMO/refs/heads/main/install.sh | bash || {
      log_warning "OSMO CLI installation failed. Install manually later."
      return
    }
  fi

  log_success "OSMO CLI installed."
}

##############################################################################
# Verification
##############################################################################

verify_deployment() {
  log_info "Cluster nodes:"
  kubectl get nodes -o wide 2>/dev/null || true
  echo ""

  log_info "Pods in namespace '${OSMO_NAMESPACE}':"
  kubectl get pods --namespace "${OSMO_NAMESPACE}" 2>/dev/null || true
  echo ""

  log_info "Services in namespace '${OSMO_NAMESPACE}':"
  kubectl get services --namespace "${OSMO_NAMESPACE}" 2>/dev/null || true
  echo ""

  wait_for_pods "${OSMO_NAMESPACE}" 300
}

##############################################################################
# Completion Message
##############################################################################

print_complete_message() {
  local node_count
  node_count="$(kubectl get nodes --no-headers 2>/dev/null | wc -l || echo 1)"

  cat <<EOF

==============================================================================
            NVIDIA OSMO — Local Deployment Complete!
==============================================================================

  Cluster: ${node_count} node(s)
$(kubectl get nodes 2>/dev/null | sed 's/^/  /')

  OSMO Web UI:
    http://${OSMO_HOSTNAME}  (port 30080)
    or: kubectl port-forward svc/osmo-ui 3000:80 -n ${OSMO_NAMESPACE}
        then visit http://localhost:3000

  OSMO API:
    kubectl port-forward svc/osmo-service 9000:80 -n ${OSMO_NAMESPACE}
    then visit http://localhost:9000/api/docs

  Login with the CLI:
    osmo login http://${OSMO_HOSTNAME} --method=dev --username=testuser

  Add more nodes later:
    bash scripts/07-add-node.sh --node-ip <IP> --node-user <USER>

  Check cluster status:
    kubectl get nodes
    kubectl get pods -n ${OSMO_NAMESPACE}

  To tear down:
    sudo bash scripts/05-cleanup.sh --config config/inventory.env

==============================================================================
EOF
}

main "$@"
