#!/usr/bin/env bash
# Submit GR00T N1.7 fine-tuning to Azure ML.
#
# Mirrors submit-azureml-openvla-oft-training.sh's resolution / submission
# patterns (Terraform outputs > env > CLI flags, idempotent environment
# registration, az ml job create with --set overrides) but targets the GR00T
# entry script and the gr00t-train.yaml command-job template.
set -o errexit -o nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../../.." && pwd))"

# shellcheck source=../../scripts/lib/common.sh
source "$REPO_ROOT/scripts/lib/common.sh"
# shellcheck source=../../scripts/lib/terraform-outputs.sh
source "$REPO_ROOT/scripts/lib/terraform-outputs.sh"
read_terraform_outputs "$REPO_ROOT/infrastructure/terraform" 2>/dev/null || true

ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

show_help() {
  cat <<'EOF'
Usage: submit-azureml-gr00t-training.sh [OPTIONS] [-- az-ml-job-flags]

Submit NVIDIA Isaac-GR00T N1.7 fine-tuning to Azure ML.

DATA SOURCE:
    -d, --dataset-name NAME      Logical dataset name (folder under DATASET_ROOT)
        --dataset-asset SPEC     AzureML data asset (NAME:VERSION or NAME@latest).
                                 Mounts the asset read-only at $DATASET_MOUNT.
        --dataset-root DIR       Container mount path (default: /workspace/data)
        --resume-from URI        Prior checkpoint folder (AzureML uri_folder URI or
                                 azureml://datastores/... path). The folder MUST
                                 contain checkpoint-N/ subdirectories; the entry
                                 script symlinks them into the output dir so
                                 HF Trainer resumes from the latest step.

PROFILES:
        --profile NAME           Apply preset (overrides individual flags below where unset):
                                   smoke-a10   - A10 24 GB pipeline validation (100 steps, batch=1)
                                   prod-a100   - 1x A100 80GB production (batch=32, 20k steps; default)

JOB IDENTITY:
    -j, --job-name NAME          Job identifier (default: gr00t-training)
        --display-name NAME      Display name override
        --experiment-name NAME   Experiment name override

GR00T RECIPE:
        --gr00t-ref REF          NVIDIA/Isaac-GR00T git ref (default: main; pin a SHA for prod)
        --base-model-path PATH   HF id or path (default: nvidia/GR00T-N1.7-3B)
        --image-key-primary K    Primary camera feature key
        --image-key-left K       Left wrist camera feature key
        --image-key-right K      Right wrist camera feature key
        --state-slices SPEC      NAME=START:END,... for state (default: right_arm=0:6,left_arm=6:12)
        --action-slices SPEC     NAME=START:END,... for action (defaults to --state-slices)
        --annotation SPEC        NAME=ORIGINAL_KEY,... for annotation (default: human.task_description=task_index)

HYPERPARAMETERS:
        --global-batch-size N    (default: 32)
        --gradient-accumulation-steps N (default: 1)
        --learning-rate F        (default: 1e-4)
        --max-steps N            (default: 20000)
        --save-steps N           (default: 2000)
        --save-total-limit N     (default: 3)
        --save-only-model BOOL   True/False (default: True; ~7GB/save vs ~19GB)
        --skip-weight-loading BOOL True/False (default: False)
        --num-gpus N             nproc-per-node for torchrun (default: 1)
        --dataloader-num-workers N (default: 4)
        --tune-projector BOOL    (default: True)
        --tune-diffusion-model BOOL (default: True)
        --tune-llm BOOL          (default: False)
        --tune-visual BOOL       (default: False)

AZUREML ASSETS:
        --environment-name NAME  AzureML env name (default: gr00t-training-env)
        --environment-version V  Env version (default: 1.0.0)
        --image IMAGE            Container image (default: nvidia/cuda:12.8.0-devel-ubuntu22.04)
        --assets-only            Register environment without submitting job

HF GATING:
        --hf-token TOKEN         HuggingFace token (required: nvidia/Cosmos-Reason2-2B is gated).
                                 Falls back to $HF_TOKEN if unset.

AZURE CONTEXT:
        --subscription-id ID
        --resource-group NAME
        --workspace-name NAME
        --compute TARGET
        --instance-type NAME     Instance type (default: gpu-a100; "" for managed compute)
        --stream                 Stream logs after submission
    -a, --save-as PATH

GENERAL:
    -h, --help
        --config-preview         Print configuration and exit

Resolution order: CLI > env vars > Terraform outputs.
EOF
}

ensure_ml_extension() {
  az extension show --name ml &>/dev/null ||
    fatal "Azure ML CLI extension not installed. Run: az extension add --name ml"
}

register_environment() {
  local name="$1" version="$2" image="$3" rg="$4" ws="$5" sub="$6"
  local env_file
  env_file=$(mktemp)

  cat >"$env_file" <<EOF
\$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: $name
version: $version
image: $image
EOF

  info "Publishing AzureML environment ${name}:${version}"
  az ml environment create --file "$env_file" \
    --name "$name" --version "$version" \
    --resource-group "$rg" --workspace-name "$ws" \
    --subscription "$sub" >/dev/null 2>&1 || \
    warn "Environment ${name}:${version} already exists or registration failed; continuing"
  rm -f "$env_file"
}

# -------- Defaults --------
environment_name="gr00t-training-env"
environment_version="1.0.0"
image="${IMAGE:-nvidia/cuda:12.8.0-devel-ubuntu22.04}"
assets_only=false

job_file="$REPO_ROOT/training/il/workflows/azureml/gr00t-train.yaml"
dataset_name="${DATASET_NAME:-schaeffler_bimanual}"
dataset_root="${DATASET_ROOT:-/workspace/data}"
dataset_asset="${DATASET_ASSET:-}"
resume_from="${RESUME_FROM:-}"

job_name="${JOB_NAME:-gr00t-training}"
display_name=""
experiment_name=""

gr00t_ref="${GR00T_REF:-main}"
base_model_path="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
image_key_primary="${IMAGE_KEY_PRIMARY:-observation.images.d405_stationary_r_0}"
image_key_left="${IMAGE_KEY_LEFT_WRIST:-observation.images.d405_stationary_l_1}"
image_key_right="${IMAGE_KEY_RIGHT_WRIST:-observation.images.d405_stationary_l_2}"
state_slices="${STATE_SLICES:-right_arm=0:6,left_arm=6:12}"
action_slices="${ACTION_SLICES:-}"
annotation_mapping="${ANNOTATION_MAPPING:-human.task_description=task_index}"

global_batch_size="${GLOBAL_BATCH_SIZE:-32}"
grad_accum_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
learning_rate="${LEARNING_RATE:-1e-4}"
max_steps="${MAX_STEPS:-20000}"
save_steps="${SAVE_STEPS:-2000}"
save_total_limit="${SAVE_TOTAL_LIMIT:-3}"
save_only_model="${SAVE_ONLY_MODEL:-True}"
skip_weight_loading="${SKIP_WEIGHT_LOADING:-False}"
num_gpus="${NUM_GPUS:-1}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-4}"
tune_projector="${TUNE_PROJECTOR:-True}"
tune_diffusion_model="${TUNE_DIFFUSION_MODEL:-True}"
tune_llm="${TUNE_LLM:-False}"
tune_visual="${TUNE_VISUAL:-False}"

hf_token="${HF_TOKEN:-}"

subscription_id="${AZURE_SUBSCRIPTION_ID:-$(get_subscription_id)}"
resource_group="${AZURE_RESOURCE_GROUP:-$(get_resource_group)}"
workspace_name="${AZUREML_WORKSPACE_NAME:-$(get_azureml_workspace)}"

compute="${AZUREML_COMPUTE:-$(get_compute_target)}"
instance_type="gpu-a100"
stream_logs=false
save_as=""
config_preview=false
forward_args=()
profile=""

# --- Profiles (applied lazily after parse) ---
apply_profile() {
  case "$1" in
    smoke-a10)
      # A10 24 GB pipeline validation: install + dataset + first save. NVIDIA
      # documents 40 GB+ as the GR00T finetune minimum, so any real training
      # step on A10 is expected to OOM. Use this profile to verify the
      # wrapper, the entry script, and HF token plumbing work end-to-end.
      [[ -z "${PROFILE_BATCH_SIZE_SET:-}" ]]              && global_batch_size=1
      [[ -z "${PROFILE_GRAD_ACCUM_SET:-}" ]]              && grad_accum_steps=16
      [[ -z "${PROFILE_MAX_STEPS_SET:-}" ]]               && max_steps=100
      [[ -z "${PROFILE_SAVE_STEPS_SET:-}" ]]              && save_steps=100
      [[ -z "${PROFILE_SAVE_TOTAL_LIMIT_SET:-}" ]]        && save_total_limit=1
      [[ -z "${PROFILE_NUM_GPUS_SET:-}" ]]                && num_gpus=1
      [[ -z "${PROFILE_INSTANCE_TYPE_SET:-}" ]]           && instance_type=gpu
      [[ -z "${PROFILE_TUNE_PROJECTOR_SET:-}" ]]          && tune_projector=False
      [[ -z "${PROFILE_TUNE_DIFFUSION_SET:-}" ]]          && tune_diffusion_model=True
      [[ -z "${PROFILE_TUNE_LLM_SET:-}" ]]                && tune_llm=False
      [[ -z "${PROFILE_TUNE_VISUAL_SET:-}" ]]             && tune_visual=False
      [[ -z "${PROFILE_SKIP_WEIGHT_LOADING_SET:-}" ]]     && skip_weight_loading=True
      [[ -z "${PROFILE_SAVE_ONLY_MODEL_SET:-}" ]]         && save_only_model=True
      [[ -z "${PROFILE_DATALOADER_WORKERS_SET:-}" ]]      && dataloader_num_workers=2
      [[ -z "${PROFILE_JOB_NAME_SET:-}" ]]                && job_name=gr00t-smoke-a10
      return 0
      ;;
    prod-a100|"")
      # 1x A100 80GB documented Quick-Start: batch=32, 20k steps, save every
      # 2k, keep last 3, save-only-model on (~7 GB/save vs ~19 GB).
      [[ -z "${PROFILE_BATCH_SIZE_SET:-}" ]]              && global_batch_size=32
      [[ -z "${PROFILE_GRAD_ACCUM_SET:-}" ]]              && grad_accum_steps=1
      [[ -z "${PROFILE_MAX_STEPS_SET:-}" ]]               && max_steps=20000
      [[ -z "${PROFILE_SAVE_STEPS_SET:-}" ]]              && save_steps=2000
      [[ -z "${PROFILE_SAVE_TOTAL_LIMIT_SET:-}" ]]        && save_total_limit=3
      [[ -z "${PROFILE_NUM_GPUS_SET:-}" ]]                && num_gpus=1
      [[ -z "${PROFILE_INSTANCE_TYPE_SET:-}" ]]           && instance_type=gpu-a100
      [[ -z "${PROFILE_SAVE_ONLY_MODEL_SET:-}" ]]         && save_only_model=True
      [[ -z "${PROFILE_DATALOADER_WORKERS_SET:-}" ]]      && dataloader_num_workers=4
      [[ -z "${PROFILE_JOB_NAME_SET:-}" ]]                && job_name=gr00t-prod-a100
      return 0
      ;;
    *)
      fatal "Unknown profile: $1 (expected smoke-a10 or prod-a100)"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)                          show_help; exit 0 ;;
    --environment-name)                 environment_name="$2"; shift 2 ;;
    --environment-version)              environment_version="$2"; shift 2 ;;
    --image|-i)                         image="$2"; shift 2 ;;
    --assets-only)                      assets_only=true; shift ;;
    --profile)                          profile="$2"; shift 2 ;;
    -d|--dataset-name)                  dataset_name="$2"; shift 2 ;;
    --dataset-asset)                    dataset_asset="$2"; shift 2 ;;
    --resume-from)                      resume_from="$2"; shift 2 ;;
    --dataset-root)                     dataset_root="$2"; shift 2 ;;
    -j|--job-name)                      job_name="$2"; PROFILE_JOB_NAME_SET=1; shift 2 ;;
    --display-name)                     display_name="$2"; shift 2 ;;
    --experiment-name)                  experiment_name="$2"; shift 2 ;;
    --gr00t-ref)                        gr00t_ref="$2"; shift 2 ;;
    --base-model-path)                  base_model_path="$2"; shift 2 ;;
    --image-key-primary)                image_key_primary="$2"; shift 2 ;;
    --image-key-left)                   image_key_left="$2"; shift 2 ;;
    --image-key-right)                  image_key_right="$2"; shift 2 ;;
    --state-slices)                     state_slices="$2"; shift 2 ;;
    --action-slices)                    action_slices="$2"; shift 2 ;;
    --annotation)                       annotation_mapping="$2"; shift 2 ;;
    --global-batch-size)                global_batch_size="$2"; PROFILE_BATCH_SIZE_SET=1; shift 2 ;;
    --gradient-accumulation-steps)      grad_accum_steps="$2"; PROFILE_GRAD_ACCUM_SET=1; shift 2 ;;
    --learning-rate)                    learning_rate="$2"; shift 2 ;;
    --max-steps)                        max_steps="$2"; PROFILE_MAX_STEPS_SET=1; shift 2 ;;
    --save-steps)                       save_steps="$2"; PROFILE_SAVE_STEPS_SET=1; shift 2 ;;
    --save-total-limit)                 save_total_limit="$2"; PROFILE_SAVE_TOTAL_LIMIT_SET=1; shift 2 ;;
    --save-only-model)                  save_only_model="$2"; PROFILE_SAVE_ONLY_MODEL_SET=1; shift 2 ;;
    --skip-weight-loading)              skip_weight_loading="$2"; PROFILE_SKIP_WEIGHT_LOADING_SET=1; shift 2 ;;
    --num-gpus)                         num_gpus="$2"; PROFILE_NUM_GPUS_SET=1; shift 2 ;;
    --dataloader-num-workers)           dataloader_num_workers="$2"; PROFILE_DATALOADER_WORKERS_SET=1; shift 2 ;;
    --tune-projector)                   tune_projector="$2"; PROFILE_TUNE_PROJECTOR_SET=1; shift 2 ;;
    --tune-diffusion-model)             tune_diffusion_model="$2"; PROFILE_TUNE_DIFFUSION_SET=1; shift 2 ;;
    --tune-llm)                         tune_llm="$2"; PROFILE_TUNE_LLM_SET=1; shift 2 ;;
    --tune-visual)                      tune_visual="$2"; PROFILE_TUNE_VISUAL_SET=1; shift 2 ;;
    --hf-token)                         hf_token="$2"; shift 2 ;;
    --subscription-id)                  subscription_id="$2"; shift 2 ;;
    --resource-group)                   resource_group="$2"; shift 2 ;;
    --workspace-name)                   workspace_name="$2"; shift 2 ;;
    --compute)                          compute="$2"; shift 2 ;;
    --instance-type)                    instance_type="$2"; PROFILE_INSTANCE_TYPE_SET=1; shift 2 ;;
    --stream)                           stream_logs=true; shift ;;
    -a|--save-as)                       save_as="$2"; shift 2 ;;
    --config-preview)                   config_preview=true; shift ;;
    --)                                 shift; forward_args=("$@"); break ;;
    *)                                  fatal "Unknown option: $1" ;;
  esac
done

[[ -n "$profile" ]] && apply_profile "$profile"

# Default action slices to the state slices if not separately specified.
[[ -z "$action_slices" ]] && action_slices="$state_slices"

# -------- Validation --------
require_tools az
ensure_ml_extension

[[ -n "$subscription_id" ]] || fatal "AZURE_SUBSCRIPTION_ID required"
[[ -n "$resource_group" ]] || fatal "AZURE_RESOURCE_GROUP required"
[[ -n "$workspace_name" ]] || fatal "AZUREML_WORKSPACE_NAME required"

if [[ -z "$dataset_asset" ]]; then
  warn "No --dataset-asset provided; entry script will fall back to ${dataset_root}/${dataset_name}"
fi

display_name="${display_name:-${job_name}-$(date +%Y%m%d-%H%M%S)}"
experiment_name="${experiment_name:-gr00t-training}"

if [[ "$config_preview" == "true" ]]; then
  section "Submission Configuration"
  print_kv "Job name" "$job_name"
  print_kv "Display name" "$display_name"
  print_kv "Experiment" "$experiment_name"
  print_kv "Compute" "$compute"
  print_kv "Instance type" "$instance_type"
  print_kv "Image" "$image"
  print_kv "Dataset name" "$dataset_name"
  print_kv "Dataset asset" "${dataset_asset:-<none>}"
  print_kv "Profile" "${profile:-<none>}"
  print_kv "GR00T ref" "$gr00t_ref"
  print_kv "Base model" "$base_model_path"
  print_kv "HF_TOKEN set" "$([[ -n "$hf_token" ]] && echo yes || echo no)"
  print_kv "num_gpus" "$num_gpus"
  print_kv "global_batch_size" "$global_batch_size"
  print_kv "gradient_accumulation_steps" "$grad_accum_steps"
  print_kv "learning_rate" "$learning_rate"
  print_kv "max_steps" "$max_steps"
  print_kv "save_steps/limit" "$save_steps/$save_total_limit"
  print_kv "save_only_model" "$save_only_model"
  print_kv "skip_weight_loading" "$skip_weight_loading"
  print_kv "tune projector/diffusion/llm/visual" "$tune_projector/$tune_diffusion_model/$tune_llm/$tune_visual"
  print_kv "state_slices" "$state_slices"
  print_kv "action_slices" "$action_slices"
  exit 0
fi

if [[ -z "$hf_token" ]]; then
  warn "HF_TOKEN not set; nvidia/Cosmos-Reason2-2B fetch will fail with 401 unless --skip-weight-loading is also True"
fi

# -------- Register environment --------
register_environment "$environment_name" "$environment_version" "$image" \
  "$resource_group" "$workspace_name" "$subscription_id"

if [[ "$assets_only" == "true" ]]; then
  info "Environment registered. Skipping job submission (--assets-only)."
  exit 0
fi

# -------- Submit job --------
submit_args=(
  --file "$job_file"
  --resource-group "$resource_group"
  --workspace-name "$workspace_name"
  --subscription "$subscription_id"
  --name "$job_name"
  --set "display_name=$display_name"
  --set "experiment_name=$experiment_name"
  --set "compute=azureml:$compute"
  --set "environment=azureml:${environment_name}:${environment_version}"
  --set "inputs.gr00t_ref=$gr00t_ref"
  --set "inputs.base_model_path=$base_model_path"
  --set "inputs.dataset_name=$dataset_name"
  --set "inputs.dataset_root=$dataset_root"
  --set "inputs.image_key_primary=$image_key_primary"
  --set "inputs.image_key_left_wrist=$image_key_left"
  --set "inputs.image_key_right_wrist=$image_key_right"
  --set "inputs.state_slices=$state_slices"
  --set "inputs.action_slices=$action_slices"
  --set "inputs.annotation_mapping=$annotation_mapping"
  --set "inputs.global_batch_size=$global_batch_size"
  --set "inputs.gradient_accumulation_steps=$grad_accum_steps"
  --set "inputs.learning_rate=$learning_rate"
  --set "inputs.max_steps=$max_steps"
  --set "inputs.save_steps=$save_steps"
  --set "inputs.save_total_limit=$save_total_limit"
  --set "inputs.save_only_model=$save_only_model"
  --set "inputs.skip_weight_loading=$skip_weight_loading"
  --set "inputs.num_gpus=$num_gpus"
  --set "inputs.dataloader_num_workers=$dataloader_num_workers"
  --set "inputs.tune_projector=$tune_projector"
  --set "inputs.tune_diffusion_model=$tune_diffusion_model"
  --set "inputs.tune_llm=$tune_llm"
  --set "inputs.tune_visual=$tune_visual"
  --set "command=bash il/scripts/gr00t/azureml-train-entry.sh"
  # Inputs flow to the entry script via env vars (the AzureML K8s extension does
  # not substitute \${{inputs.X}} placeholders inside `command:` for Download mounts).
  --set "environment_variables.GR00T_REF=$gr00t_ref"
  --set "environment_variables.BASE_MODEL_PATH=$base_model_path"
  --set "environment_variables.DATASET_NAME=$dataset_name"
  --set "environment_variables.DATASET_ROOT=$dataset_root"
  --set "environment_variables.IMAGE_KEY_PRIMARY=$image_key_primary"
  --set "environment_variables.IMAGE_KEY_LEFT_WRIST=$image_key_left"
  --set "environment_variables.IMAGE_KEY_RIGHT_WRIST=$image_key_right"
  --set "environment_variables.STATE_SLICES=$state_slices"
  --set "environment_variables.ACTION_SLICES=$action_slices"
  --set "environment_variables.ANNOTATION_MAPPING=$annotation_mapping"
  --set "environment_variables.GLOBAL_BATCH_SIZE=$global_batch_size"
  --set "environment_variables.GRADIENT_ACCUMULATION_STEPS=$grad_accum_steps"
  --set "environment_variables.LEARNING_RATE=$learning_rate"
  --set "environment_variables.MAX_STEPS=$max_steps"
  --set "environment_variables.SAVE_STEPS=$save_steps"
  --set "environment_variables.SAVE_TOTAL_LIMIT=$save_total_limit"
  --set "environment_variables.SAVE_ONLY_MODEL=$save_only_model"
  --set "environment_variables.SKIP_WEIGHT_LOADING=$skip_weight_loading"
  --set "environment_variables.NUM_GPUS=$num_gpus"
  --set "environment_variables.DATALOADER_NUM_WORKERS=$dataloader_num_workers"
  --set "environment_variables.TUNE_PROJECTOR=$tune_projector"
  --set "environment_variables.TUNE_DIFFUSION_MODEL=$tune_diffusion_model"
  --set "environment_variables.TUNE_LLM=$tune_llm"
  --set "environment_variables.TUNE_VISUAL=$tune_visual"
  --set "environment_variables.AZURE_SUBSCRIPTION_ID=$subscription_id"
  --set "environment_variables.AZURE_RESOURCE_GROUP=$resource_group"
  --set "environment_variables.AZUREML_WORKSPACE_NAME=$workspace_name"
)

# Only attach instance_type when set; managed compute clusters do not accept
# a Kubernetes InstanceType CRD reference.
[[ -n "$instance_type" ]] && submit_args+=( --set "resources.instance_type=$instance_type" )

# HF token plumbed through environment_variables (never recorded in inputs to
# avoid leaking via lineage). nvidia/Cosmos-Reason2-2B is gated.
[[ -n "$hf_token" ]] && submit_args+=( --set "environment_variables.HF_TOKEN=$hf_token" )

# Mount a registered AzureML data asset (uri_folder) as `dataset_asset` input and
# export its mount path as DATASET_MOUNT for the entry script.
if [[ -n "$dataset_asset" ]]; then
  [[ "$dataset_asset" == azureml:* ]] || dataset_asset="azureml:$dataset_asset"
  submit_args+=(
    --set "inputs.dataset_asset.type=uri_folder"
    --set "inputs.dataset_asset.mode=ro_mount"
    --set "inputs.dataset_asset.path=$dataset_asset"
    --set 'environment_variables.DATASET_MOUNT=${{inputs.dataset_asset}}'
  )
fi

# Mount a prior checkpoint folder as `resume_checkpoint` input and surface its
# mount path as RESUME_CHECKPOINT. Accepts either a bare azureml:// URI or an
# `azureml:NAME:VERSION` data-asset reference (same shape as --dataset-asset).
if [[ -n "$resume_from" ]]; then
  if [[ "$resume_from" != azureml:* && "$resume_from" != azureml://* ]]; then
    resume_from="azureml:$resume_from"
  fi
  submit_args+=(
    --set "inputs.resume_checkpoint.type=uri_folder"
    --set "inputs.resume_checkpoint.mode=ro_mount"
    --set "inputs.resume_checkpoint.path=$resume_from"
    --set 'environment_variables.RESUME_CHECKPOINT=${{inputs.resume_checkpoint}}'
  )
fi

[[ "$stream_logs" == "true" ]] && submit_args+=(--stream)
[[ -n "$save_as" ]] && submit_args+=(--save-as "$save_as")

info "Submitting job ${job_name} to compute=${compute} (${instance_type:-<managed>})"
az ml job create "${submit_args[@]}" "${forward_args[@]}"

section "Deployment Summary"
print_kv "Job" "$job_name"
print_kv "Display name" "$display_name"
print_kv "Compute" "$compute"
print_kv "Instance type" "${instance_type:-<managed>}"
print_kv "Environment" "${environment_name}:${environment_version}"
print_kv "Profile" "${profile:-<none>}"
print_kv "Dataset" "${dataset_asset:-${dataset_root}/${dataset_name}}"
print_kv "Resume from" "${resume_from:-<none>}"
print_kv "Base model" "$base_model_path"
print_kv "GR00T ref" "$gr00t_ref"
print_kv "global_batch_size / steps" "$global_batch_size / $max_steps"
