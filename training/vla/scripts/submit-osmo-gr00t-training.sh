#!/usr/bin/env bash
# Submit a GR00T-N1.5-3B VLA finetune workflow to OSMO.
# Renders the workflow template with the inlined trainer script, then submits.
set -o errexit -o nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../../.." && pwd))"

source "$REPO_ROOT/scripts/lib/common.sh"

# Source .env file if present (for credentials and overrides).
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

show_help() {
  cat << 'EOF'
Usage: submit-osmo-gr00t-training.sh [OPTIONS] [-- osmo-submit-flags]

Submit a GR00T-N1.5-3B Vision-Language-Action finetune workflow to OSMO. The
selected trainer script is inlined into the workflow template at the
`__TRAIN_SCRIPT__` placeholder before submission.

EMBODIMENT:
    -e, --embodiment NAME     Trainer variant (default: dual_arm)
                                dual_arm  -> train_gr00t_dual_arm.py
                                n1_5_3b   -> train_gr00t_dual_arm_n1_5_3b.py
        --train-script PATH   Explicit trainer .py to inline (overrides -e)

TRAINING OPTIONS:
    -w, --workflow PATH       Workflow template (default: gr00t-train.yaml)
    -j, --job-name NAME       OSMO workflow name
        --base-model ID       HuggingFace model id or local path
        --max-steps N         Total training steps
        --save-steps N        Checkpoint save frequency
        --batch-size N        Training batch size
        --grad-accum N        Gradient accumulation steps
        --lr LR               Learning rate
        --num-workers N       Dataloader worker processes
        --resume-from PATH    Checkpoint path to resume from

DATA SOURCE (in-cluster LocalStack S3):
        --dataset-name NAME       Dataset short name
        --dataset-mount-path P    Local mount path inside the pod
        --dataset-s3-uri URI      s3:// URI to sync from
        --s3-endpoint-url URL     LocalStack S3 endpoint
        --s3-access-key KEY       S3 access key
        --s3-secret-key KEY       S3 secret key

RESOURCES:
        --image IMAGE         Container image
        --platform NAME       GPU node platform (e.g. gpu_5090)
        --gpu N / --cpu N / --memory SIZE / --storage SIZE
        --hf-token TOKEN      HuggingFace token for gated models
        --pool NAME / --priority N

OTHER:
        --no-follow           Submit and detach (do not tail logs)
        --use-local-osmo      Use the local osmo-dev CLI instead of osmo
        --config-preview      Print configuration and exit
    -h, --help                Show this help message

Values resolved: CLI > environment variables > template defaults.
Additional arguments after -- are forwarded to osmo workflow submit.
EOF
}

#------------------------------------------------------------------------------
# Defaults
#------------------------------------------------------------------------------

workflow="$REPO_ROOT/training/vla/workflows/osmo/gr00t-train.yaml"
embodiment="${EMBODIMENT:-dual_arm}"
train_script="${TRAIN_SCRIPT:-}"

job_name="${WORKFLOW_NAME:-}"
base_model="${BASE_MODEL:-}"
max_steps="${MAX_STEPS:-}"
save_steps="${SAVE_STEPS:-}"
batch_size="${BATCH_SIZE:-}"
grad_accum="${GRAD_ACCUM:-}"
lr="${LR:-}"
num_workers="${NUM_WORKERS:-}"
resume_from="${RESUME_FROM:-}"

dataset_name="${DATASET_NAME:-}"
dataset_mount_path="${DATASET_MOUNT_PATH:-}"
dataset_s3_uri="${DATASET_S3_URI:-}"
s3_endpoint_url="${S3_ENDPOINT_URL:-}"
s3_access_key="${S3_ACCESS_KEY:-}"
s3_secret_key="${S3_SECRET_KEY:-}"

image="${IMAGE:-}"
platform="${PLATFORM:-}"
gpu="${GPU:-}"
cpu="${CPU:-}"
memory="${MEMORY:-}"
storage="${STORAGE:-}"
hf_token="${HF_TOKEN:-}"
pool="${POOL:-}"
priority="${PRIORITY:-}"

follow=true
use_local_osmo=false
config_preview=false
forward_args=()

TMP_DIR="$SCRIPT_DIR/.tmp"
RENDERED="$TMP_DIR/gr00t-train.rendered.yaml"

#------------------------------------------------------------------------------
# Parse Arguments
#------------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)              show_help; exit 0 ;;
    -e|--embodiment)        embodiment="$2"; shift 2 ;;
    --train-script)         train_script="$2"; shift 2 ;;
    -w|--workflow)          workflow="$2"; shift 2 ;;
    -j|--job-name)          job_name="$2"; shift 2 ;;
    --base-model)           base_model="$2"; shift 2 ;;
    --max-steps)            max_steps="$2"; shift 2 ;;
    --save-steps)           save_steps="$2"; shift 2 ;;
    --batch-size)           batch_size="$2"; shift 2 ;;
    --grad-accum)           grad_accum="$2"; shift 2 ;;
    --lr)                   lr="$2"; shift 2 ;;
    --num-workers)          num_workers="$2"; shift 2 ;;
    --resume-from)          resume_from="$2"; shift 2 ;;
    --dataset-name)         dataset_name="$2"; shift 2 ;;
    --dataset-mount-path)   dataset_mount_path="$2"; shift 2 ;;
    --dataset-s3-uri)       dataset_s3_uri="$2"; shift 2 ;;
    --s3-endpoint-url)      s3_endpoint_url="$2"; shift 2 ;;
    --s3-access-key)        s3_access_key="$2"; shift 2 ;;
    --s3-secret-key)        s3_secret_key="$2"; shift 2 ;;
    --image)                image="$2"; shift 2 ;;
    --platform)             platform="$2"; shift 2 ;;
    --gpu)                  gpu="$2"; shift 2 ;;
    --cpu)                  cpu="$2"; shift 2 ;;
    --memory)               memory="$2"; shift 2 ;;
    --storage)              storage="$2"; shift 2 ;;
    --hf-token)             hf_token="$2"; shift 2 ;;
    --pool)                 pool="$2"; shift 2 ;;
    --priority)             priority="$2"; shift 2 ;;
    --no-follow)            follow=false; shift ;;
    --use-local-osmo)       use_local_osmo=true; shift ;;
    --config-preview)       config_preview=true; shift ;;
    --)                     shift; forward_args=("$@"); break ;;
    *)                      forward_args+=("$1"); shift ;;
  esac
done

#------------------------------------------------------------------------------
# Resolve trainer script from embodiment
#------------------------------------------------------------------------------

if [[ -z "$train_script" ]]; then
  case "$embodiment" in
    dual_arm) train_script="$SCRIPT_DIR/train_gr00t_dual_arm.py" ;;
    n1_5_3b)  train_script="$SCRIPT_DIR/train_gr00t_dual_arm_n1_5_3b.py" ;;
    *)        fatal "Unknown embodiment: $embodiment (use: dual_arm, n1_5_3b, or --train-script)" ;;
  esac
fi

#------------------------------------------------------------------------------
# Validation
#------------------------------------------------------------------------------

[[ "$use_local_osmo" == "true" ]] && activate_local_osmo

require_tools osmo awk

[[ -f "$workflow" ]] || fatal "Workflow template not found: $workflow"
[[ -f "$train_script" ]] || fatal "Trainer script not found: $train_script"
grep -q '__TRAIN_SCRIPT__' "$workflow" || fatal "Template missing __TRAIN_SCRIPT__ placeholder: $workflow"

if [[ "$config_preview" == "true" ]]; then
  section "Configuration Preview"
  print_kv "Embodiment" "$embodiment"
  print_kv "Trainer" "$train_script"
  print_kv "Workflow" "$workflow"
  print_kv "Job Name" "${job_name:-<template default>}"
  print_kv "Base Model" "${base_model:-<template default>}"
  print_kv "Max Steps" "${max_steps:-<template default>}"
  print_kv "Save Steps" "${save_steps:-<template default>}"
  print_kv "Batch Size" "${batch_size:-<template default>}"
  print_kv "Learning Rate" "${lr:-<template default>}"
  print_kv "Dataset S3 URI" "${dataset_s3_uri:-<template default>}"
  print_kv "Image" "${image:-<template default>}"
  print_kv "Platform" "${platform:-<template default>}"
  print_kv "Follow Logs" "$follow"
  exit 0
fi

#------------------------------------------------------------------------------
# Render Workflow (inline the trainer at __TRAIN_SCRIPT__, indented 12 spaces)
#------------------------------------------------------------------------------

info "Rendering workflow with inlined trainer..."
mkdir -p "$TMP_DIR"
indented="$(awk '{print "            " $0}' "$train_script")"
awk -v body="$indented" '{ if ($0 ~ /__TRAIN_SCRIPT__/) { print body } else { print } }' \
  "$workflow" > "$RENDERED" || fatal "Failed to render workflow"

#------------------------------------------------------------------------------
# Build --set list (only non-empty overrides)
#------------------------------------------------------------------------------

sets=()
add_set() { [[ -z "$2" ]] || sets+=("$1=$2"); }
add_set workflow_name      "$job_name"
add_set base_model         "$base_model"
add_set max_steps          "$max_steps"
add_set save_steps         "$save_steps"
add_set batch_size         "$batch_size"
add_set grad_accum         "$grad_accum"
add_set lr                 "$lr"
add_set num_workers        "$num_workers"
add_set resume_from        "$resume_from"
add_set dataset_name       "$dataset_name"
add_set dataset_mount_path "$dataset_mount_path"
add_set dataset_s3_uri     "$dataset_s3_uri"
add_set s3_endpoint_url    "$s3_endpoint_url"
add_set s3_access_key      "$s3_access_key"
add_set s3_secret_key      "$s3_secret_key"
add_set image              "$image"
add_set platform           "$platform"
add_set gpu                "$gpu"
add_set cpu                "$cpu"
add_set memory             "$memory"
add_set storage            "$storage"
add_set hf_token           "$hf_token"

submit_args=(workflow submit "$RENDERED")
[[ ${#sets[@]} -gt 0 ]] && submit_args+=(--set "${sets[@]}")
[[ -n "$pool" ]] && submit_args+=(--pool "$pool")
[[ -n "$priority" ]] && submit_args+=(--priority "$priority")
[[ "$follow" == "false" ]] && submit_args+=(--no-follow)
[[ ${#forward_args[@]} -gt 0 ]] && submit_args+=("${forward_args[@]}")

#------------------------------------------------------------------------------
# Submit Workflow
#------------------------------------------------------------------------------

info "Submitting GR00T training workflow to OSMO..."
info "  Embodiment: $embodiment"
info "  Trainer:    $(basename "$train_script")"
info "  Workflow:   $(basename "$workflow")"
osmo "${submit_args[@]}" || fatal "Failed to submit workflow"

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
section "Deployment Summary"
print_kv "Embodiment" "$embodiment"
print_kv "Trainer" "$(basename "$train_script")"
print_kv "Job Name" "${job_name:-<template default>}"
print_kv "Base Model" "${base_model:-<template default>}"
print_kv "Max Steps" "${max_steps:-<template default>}"
print_kv "Dataset S3 URI" "${dataset_s3_uri:-<template default>}"
print_kv "Workflow" "$workflow"

info "Workflow submitted successfully"
