#!/usr/bin/env bash
# AzureML entrypoint for LeRobot training jobs submitted by submit-azureml-lerobot-training.sh.
# Uploaded as part of the code asset; cwd inside the container is the contents of training/.
set -euo pipefail

echo "=== LeRobot AzureML Training ==="

# Restore `training/` prefix so absolute references (training/il/...) and python -m
# `training.il.scripts...` resolve when cwd is the contents of training/.
if [[ ! -e training ]]; then ln -s . training; fi

# Install runtime dependencies from pre-compiled requirements
apt-get update -qq && apt-get install -y -qq ffmpeg git build-essential >/dev/null 2>&1
pip install --quiet uv

LEROBOT_REQUIREMENTS="training/il/lerobot/requirements.txt"
if [[ ! -f "${LEROBOT_REQUIREMENTS}" ]]; then
  echo "ERROR: LeRobot requirements not found at ${LEROBOT_REQUIREMENTS}" >&2
  exit 1
fi
uv pip install --system --requirement "${LEROBOT_REQUIREMENTS}"

# Build lerobot-train args
train_args=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --policy.type="${POLICY_TYPE}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --policy.device=cuda
)

# Resolve data source: Azure Blob Storage when STORAGE_ACCOUNT is set, otherwise HuggingFace Hub
if [[ -n "${STORAGE_ACCOUNT:-}" ]]; then
  echo "Downloading dataset from Azure Blob Storage (${STORAGE_ACCOUNT}/${STORAGE_CONTAINER}/${BLOB_PREFIX})..."
  python3 -m training.il.scripts.lerobot.download_dataset
  FULL_DATASET_PATH="${DATASET_ROOT}/${DATASET_REPO_ID}"
  echo "Dataset materialized at: ${FULL_DATASET_PATH}"
  train_args+=(
    --dataset.root="${FULL_DATASET_PATH}"
    --dataset.use_imagenet_stats=false
    --dataset.video_backend=pyav
  )
elif [[ -n "${HF_TOKEN:-}" ]]; then
  python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
fi

if [[ "${WANDB_ENABLE:-true}" == "true" ]]; then
  train_args+=(--wandb.enable=true)
  [[ -n "${WANDB_PROJECT:-}" ]] && train_args+=(--wandb.project="${WANDB_PROJECT}")
else
  train_args+=(--wandb.enable=false)
fi

[[ -n "${POLICY_REPO_ID:-}" ]] && train_args+=(--policy.repo_id="${POLICY_REPO_ID}")
[[ -n "${TRAINING_STEPS:-}" ]] && train_args+=(--steps="${TRAINING_STEPS}")
[[ -n "${BATCH_SIZE:-}" ]] && train_args+=(--batch_size="${BATCH_SIZE}")
[[ -n "${EVAL_FREQ:-}" ]] && train_args+=(--eval_freq="${EVAL_FREQ}")
[[ -n "${SAVE_FREQ:-}" ]] && train_args+=(--save_freq="${SAVE_FREQ}")

echo "Running: lerobot-train ${train_args[*]}"
lerobot-train "${train_args[@]}"

echo "=== Training Complete ==="

if [[ -n "${REGISTER_CHECKPOINT:-}" ]]; then
  echo "Registering checkpoint to Azure ML..."
  python3 -m training.il.scripts.lerobot.register_checkpoint
fi
