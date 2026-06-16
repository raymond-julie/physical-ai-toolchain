#!/usr/bin/env bash
# Bootstrap FluxCD on a target Kubernetes cluster for fleet deployment
set -o errexit -o nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../.." && pwd))"
# shellcheck source=../../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"

DEFAULT_CLUSTER="tegra"
DEFAULT_GIT_URL="ssh://git@github.com/<your-org>/physical-ai-toolchain"
DEFAULT_GIT_BRANCH="main"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Bootstrap FluxCD on a target Kubernetes cluster for fleet deployment.

Two modes:
  flux bootstrap (default) installs Flux and commits the gotk-* manifests back to
  your Git repository, regenerating flux-system/gotk-sync.yaml with your --url and
  --path. Requires the flux CLI and write access to the repo.

  --apply applies the manifests already committed in this repo with
  'kubectl apply -k <cluster>/flux-system' (no commit back). Use when the gotk-*
  files are already correct for your fork.

OPTIONS:
    -h, --help               Show this help message
    -c, --cluster NAME       Cluster overlay under clusters/ (default: $DEFAULT_CLUSTER)
    -u, --url URL            Git repository URL (default: placeholder, must be set
                             unless --apply)
    -b, --branch BRANCH      Git branch (default: $DEFAULT_GIT_BRANCH)
    -p, --path PATH          Flux sync path (default: ./fleet-deployment/gitops/clusters/<cluster>)
    --apply                  Apply committed manifests via kubectl instead of flux bootstrap
    --config-preview         Print configuration and exit

EXAMPLES:
    $(basename "$0") --url ssh://git@github.com/acme/physical-ai-toolchain
    $(basename "$0") --apply
    $(basename "$0") --config-preview
EOF
}

# Defaults
cluster="$DEFAULT_CLUSTER"
git_url="$DEFAULT_GIT_URL"
git_branch="$DEFAULT_GIT_BRANCH"
flux_path=""
apply_only=false
config_preview=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)         show_help; exit 0 ;;
    -c|--cluster)      cluster="$2"; shift 2 ;;
    -u|--url)          git_url="$2"; shift 2 ;;
    -b|--branch)       git_branch="$2"; shift 2 ;;
    -p|--path)         flux_path="$2"; shift 2 ;;
    --apply)           apply_only=true; shift ;;
    --config-preview)  config_preview=true; shift ;;
    *)                 fatal "Unknown option: $1" ;;
  esac
done

require_tools kubectl

#------------------------------------------------------------------------------
# Gather Configuration
#------------------------------------------------------------------------------

cluster_dir="$SCRIPT_DIR/clusters/$cluster"
flux_system_dir="$cluster_dir/flux-system"
[[ -z "$flux_path" ]] && flux_path="./fleet-deployment/gitops/clusters/$cluster"

[[ -d "$flux_system_dir" ]] || fatal "Cluster overlay not found: $flux_system_dir"

mode="$([[ "$apply_only" == "true" ]] && echo 'kubectl apply -k' || echo 'flux bootstrap git')"

if [[ "$config_preview" == "true" ]]; then
  section "Configuration Preview"
  print_kv "Cluster" "$cluster"
  print_kv "Overlay" "$flux_system_dir"
  print_kv "Mode" "$mode"
  print_kv "Git URL" "$([[ "$apply_only" == "true" ]] && echo 'n/a (apply mode)' || echo "$git_url")"
  print_kv "Git branch" "$git_branch"
  print_kv "Flux path" "$flux_path"
  exit 0
fi

#------------------------------------------------------------------------------
# FluxCD Bootstrap
#------------------------------------------------------------------------------
section "FluxCD Bootstrap"

if [[ "$apply_only" == "true" ]]; then
  info "Applying committed Flux manifests for cluster '$cluster'"
  warn "Ensure the flux-system SSH Secret and image/blob pull Secrets exist first"
  kubectl apply -k "$flux_system_dir"
else
  [[ "$git_url" == *"<your-org>"* ]] && fatal "Set --url to your repository (gotk-sync.yaml ships a placeholder URL)"
  require_tools flux
  info "Running Flux pre-flight checks"
  flux check --pre
  info "Bootstrapping Flux against $git_url ($git_branch), path $flux_path"
  flux bootstrap git --url="$git_url" --branch="$git_branch" --path="$flux_path"
fi

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
section "Summary"
print_kv "Cluster" "$cluster"
print_kv "Mode" "$mode"
print_kv "Flux path" "$flux_path"
info "Bootstrap complete — check 'flux get kustomizations' to watch reconciliation"
