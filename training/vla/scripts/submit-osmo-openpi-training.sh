#!/usr/bin/env bash
# Submit an openpi (pi0 / pi0.5) VLA finetune workflow to OSMO.
# Renders the workflow template with the inlined policy + trainer scripts,
# then submits.
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
Usage: submit-osmo-openpi-training.sh [OPTIONS] [-- osmo-submit-flags]

Submit an openpi (pi0 / pi0.5) Vision-Language-Action finetune workflow to OSMO.
The selected policy module and trainer script are inlined into the workflow
template at the `__POLICY_SCRIPT__` and `__TRAIN_SCRIPT__` placeholders.

EMBODIMENT:
    -e, --embodiment NAME     Policy/trainer pair (default: ur5e_dual)
                                ur5e_dual -> openpi_ur5e_dual_arm_policy.py
                                             + train_openpi_ur5e_dual_arm.py
                                ur10e     -> openpi_ur10e_policy.py
                                             + train_openpi_ur10e.py
        --policy-script PATH  Explicit policy .py to inline (overrides -e)
        --train-script PATH   Explicit trainer .py to inline (overrides -e)

MODEL:
        --model-variant V     pi05 (default) | pi0
        --train-mode MODE     lora (default) | full

TRAINING OPTIONS:
    -w, --workflow PATH       Workflow template (default: openpi-train.yaml)
    -j, --job-name NAME       OSMO workflow name
        --exp-name NAME       openpi experiment name
        --openpi-ref REF      openpi git ref to clone
        --max-steps N         Total training steps
        --save-interval N     Checkpoint save interval
        --batch-size N        Training batch size
        --num-workers N       Dataloader worker processes
        --fsdp-devices N      Shard model across N devices
        --default-prompt STR  Fallback prompt when tasks.jsonl is missing
        --resume              Resume from the latest checkpoint

DATA SOURCE (in-cluster LocalStack S3):
        --dataset-name NAME / --dataset-mount-path P / --dataset-s3-uri URI
        --s3-endpoint-url URL / --s3-access-key KEY / --s3-secret-key KEY

RESOURCES:
        --image IMAGE / --platform NAME / --gpu N / --cpu N
        --memory SIZE / --storage SIZE / --hf-token TOKEN
        --pool NAME / --priority N

LOGGING:
        --wandb-api-key KEY / --wandb-project NAME / --wandb-disabled BOOL

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

workflow="$REPO_ROOT/training/vla/workflows/osmo/openpi-train.yaml"
embodiment="${EMBODIMENT:-ur5e_dual}"
policy_script="${POLICY_PY:-}"
train_script="${TRAIN_SCRIPT:-}"

model_variant="${MODEL_VARIANT:-}"
train_mode="${TRAIN_MODE:-}"

job_name="${WORKFLOW_NAME:-}"
exp_name="${EXP_NAME:-}"
openpi_ref="${OPENPI_REF:-}"
max_steps="${MAX_STEPS:-}"
save_interval="${SAVE_INTERVAL:-}"
batch_size="${BATCH_SIZE:-}"
num_workers="${NUM_WORKERS:-}"
fsdp_devices="${FSDP_DEVICES:-}"
default_prompt="${DEFAULT_PROMPT:-}"
resume="${RESUME:-}"

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
wandb_api_key="${WANDB_API_KEY:-}"
wandb_project="${WANDB_PROJECT:-}"
wandb_disabled="${WANDB_DISABLED:-}"
pool="${POOL:-}"
priority="${PRIORITY:-}"

follow=true
use_local_osmo=false
config_preview=false
forward_args=()

TMP_DIR="$SCRIPT_DIR/.tmp"
RENDERED="$TMP_DIR/openpi-train.rendered.yaml"

#------------------------------------------------------------------------------
# Parse Arguments
#------------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)              show_help; exit 0 ;;
    -e|--embodiment)        embodiment="$2"; shift 2 ;;
    --policy-script)        policy_script="$2"; shift 2 ;;
    --train-script)         train_script="$2"; shift 2 ;;
    --model-variant)        model_variant="$2"; shift 2 ;;
    --train-mode)           train_mode="$2"; shift 2 ;;
    -w|--workflow)          workflow="$2"; shift 2 ;;
    -j|--job-name)          job_name="$2"; shift 2 ;;
    --exp-name)             exp_name="$2"; shift 2 ;;
    --openpi-ref)           openpi_ref="$2"; shift 2 ;;
    --max-steps)            max_steps="$2"; shift 2 ;;
    --save-interval)        save_interval="$2"; shift 2 ;;
    --batch-size)           batch_size="$2"; shift 2 ;;
    --num-workers)          num_workers="$2"; shift 2 ;;
    --fsdp-devices)         fsdp_devices="$2"; shift 2 ;;
    --default-prompt)       default_prompt="$2"; shift 2 ;;
    --resume)               resume="true"; shift ;;
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
    --wandb-api-key)        wandb_api_key="$2"; shift 2 ;;
    --wandb-project)        wandb_project="$2"; shift 2 ;;
    --wandb-disabled)       wandb_disabled="$2"; shift 2 ;;
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
# Resolve policy + trainer scripts from embodiment
#------------------------------------------------------------------------------

if [[ -z "$policy_script" || -z "$train_script" ]]; then
  case "$embodiment" in
    ur5e_dual)
      policy_script="${policy_script:-$SCRIPT_DIR/openpi_ur5e_dual_arm_policy.py}"
      train_script="${train_script:-$SCRIPT_DIR/train_openpi_ur5e_dual_arm.py}"
      ;;
    ur10e)
      policy_script="${policy_script:-$SCRIPT_DIR/openpi_ur10e_policy.py}"
      train_script="${train_script:-$SCRIPT_DIR/train_openpi_ur10e.py}"
      ;;
    *)
      fatal "Unknown embodiment: $embodiment (use: ur5e_dual, ur10e, or --policy-script/--train-script)"
      ;;
  esac
fi

#------------------------------------------------------------------------------
# Validation
#------------------------------------------------------------------------------

[[ "$use_local_osmo" == "true" ]] && activate_local_osmo

require_tools osmo awk

[[ -f "$workflow" ]] || fatal "Workflow template not found: $workflow"
[[ -f "$policy_script" ]] || fatal "Policy script not found: $policy_script"
[[ -f "$train_script" ]] || fatal "Trainer script not found: $train_script"
grep -q '__POLICY_SCRIPT__' "$workflow" || fatal "Template missing __POLICY_SCRIPT__ placeholder: $workflow"
grep -q '__TRAIN_SCRIPT__' "$workflow" || fatal "Template missing __TRAIN_SCRIPT__ placeholder: $workflow"

if [[ -n "$model_variant" ]]; then
  case "$model_variant" in pi05|pi0) ;; *) fatal "Invalid --model-variant: $model_variant (use pi05, pi0)" ;; esac
fi
if [[ -n "$train_mode" ]]; then
  case "$train_mode" in lora|full) ;; *) fatal "Invalid --train-mode: $train_mode (use lora, full)" ;; esac
fi

if [[ "$config_preview" == "true" ]]; then
  section "Configuration Preview"
  print_kv "Embodiment" "$embodiment"
  print_kv "Policy" "$policy_script"
  print_kv "Trainer" "$train_script"
  print_kv "Workflow" "$workflow"
  print_kv "Model Variant" "${model_variant:-<template default>}"
  print_kv "Train Mode" "${train_mode:-<template default>}"
  print_kv "Job Name" "${job_name:-<template default>}"
  print_kv "Exp Name" "${exp_name:-<template default>}"
  print_kv "Max Steps" "${max_steps:-<template default>}"
  print_kv "Batch Size" "${batch_size:-<template default>}"
  print_kv "Dataset S3 URI" "${dataset_s3_uri:-<template default>}"
  print_kv "Image" "${image:-<template default>}"
  print_kv "Platform" "${platform:-<template default>}"
  print_kv "Follow Logs" "$follow"
  exit 0
fi

#------------------------------------------------------------------------------
# Render Workflow (inline policy + trainer, each indented 12 spaces)
#------------------------------------------------------------------------------

info "Rendering workflow with inlined policy + trainer..."
mkdir -p "$TMP_DIR"
indent12() { awk '{print "            " $0}' "$1"; }
policy_body="$(indent12 "$policy_script")"
train_body="$(indent12 "$train_script")"

awk -v body="$policy_body" '{ if ($0 ~ /__POLICY_SCRIPT__/) { print body } else { print } }' \
  "$workflow" > "$RENDERED.tmp" || fatal "Failed to inline policy module"
awk -v body="$train_body" '{ if ($0 ~ /__TRAIN_SCRIPT__/) { print body } else { print } }' \
  "$RENDERED.tmp" > "$RENDERED" || fatal "Failed to inline trainer"
rm -f "$RENDERED.tmp"

#------------------------------------------------------------------------------
# Build --set list (only non-empty overrides)
#------------------------------------------------------------------------------

sets=()
add_set() { [[ -z "$2" ]] || sets+=("$1=$2"); }
add_set workflow_name      "$job_name"
add_set exp_name           "$exp_name"
add_set openpi_ref         "$openpi_ref"
add_set model_variant      "$model_variant"
add_set train_mode         "$train_mode"
add_set max_steps          "$max_steps"
add_set save_interval      "$save_interval"
add_set batch_size         "$batch_size"
add_set num_workers        "$num_workers"
add_set fsdp_devices       "$fsdp_devices"
add_set default_prompt     "$default_prompt"
add_set resume             "$resume"
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
add_set wandb_api_key      "$wandb_api_key"
add_set wandb_project      "$wandb_project"
add_set wandb_disabled     "$wandb_disabled"

submit_args=(workflow submit "$RENDERED")
[[ ${#sets[@]} -gt 0 ]] && submit_args+=(--set "${sets[@]}")
[[ -n "$pool" ]] && submit_args+=(--pool "$pool")
[[ -n "$priority" ]] && submit_args+=(--priority "$priority")
[[ "$follow" == "false" ]] && submit_args+=(--no-follow)
[[ ${#forward_args[@]} -gt 0 ]] && submit_args+=("${forward_args[@]}")

#------------------------------------------------------------------------------
# Submit Workflow
#------------------------------------------------------------------------------

info "Submitting openpi training workflow to OSMO..."
info "  Embodiment: $embodiment"
info "  Policy:     $(basename "$policy_script")"
info "  Trainer:    $(basename "$train_script")"
info "  Workflow:   $(basename "$workflow")"
osmo "${submit_args[@]}" || fatal "Failed to submit workflow"

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
section "Deployment Summary"
print_kv "Embodiment" "$embodiment"
print_kv "Policy" "$(basename "$policy_script")"
print_kv "Trainer" "$(basename "$train_script")"
print_kv "Model Variant" "${model_variant:-<template default>}"
print_kv "Train Mode" "${train_mode:-<template default>}"
print_kv "Job Name" "${job_name:-<template default>}"
print_kv "Dataset S3 URI" "${dataset_s3_uri:-<template default>}"
print_kv "Workflow" "$workflow"

info "Workflow submitted successfully"
