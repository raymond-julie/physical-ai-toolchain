#!/usr/bin/env bash
# AzureML entrypoint for GR00T N1.7 fine-tuning jobs.
#
# Cwd in the container is the contents of training/; the code asset uploaded by
# submit-azureml-gr00t-training.sh restores the `training/` prefix via symlink
# so `python -m training.il.scripts.gr00t.*` resolves.
#
# Stages:
#   1. Resolve placeholders + restore training/ symlink
#   2. Install system deps (CUDA 12.8 devel image lacks ffmpeg/libegl1) + uv
#   3. Clone NVIDIA/Isaac-GR00T at the requested ref + uv sync --frozen
#   4. Copy the mounted LeRobot dataset to a writable location and convert
#      v3 -> v2.1 (idempotent) + write meta/modality.json
#   5. Copy the UR5e bimanual modality config into examples/UR5eBimanual/
#   6. Run torchrun gr00t/experiment/launch_finetune.py with the resolved flags
#
# Environment variables (set by submit-azureml-gr00t-training.sh):
#   GR00T_REF                     Git ref (sha/tag/branch) for NVIDIA/Isaac-GR00T
#   BASE_MODEL_PATH               HF id or path (default: nvidia/GR00T-N1.7-3B)
#   DATASET_NAME                  Logical dataset name (folder under dataset_root)
#   DATASET_MOUNT                 AzureML uri_folder mount path (read-only)
#   DATASET_ROOT                  Writable working directory (default: /workspace/data)
#   IMAGE_KEY_PRIMARY             LeRobot feature for primary camera
#   IMAGE_KEY_LEFT_WRIST          LeRobot feature for left wrist camera
#   IMAGE_KEY_RIGHT_WRIST         LeRobot feature for right wrist camera
#   STATE_SLICES                  Comma-separated NAME=START:END entries
#   ACTION_SLICES                 Comma-separated NAME=START:END entries
#   ANNOTATION_MAPPING            Comma-separated NAME=ORIGINAL_KEY entries (optional)
#   GLOBAL_BATCH_SIZE             int
#   GRADIENT_ACCUMULATION_STEPS   int
#   LEARNING_RATE                 float
#   MAX_STEPS                     int
#   SAVE_STEPS                    int
#   SAVE_TOTAL_LIMIT              int
#   SAVE_ONLY_MODEL               True/False
#   SKIP_WEIGHT_LOADING           True/False
#   NUM_GPUS                      int
#   DATALOADER_NUM_WORKERS        int
#   TUNE_PROJECTOR                True/False
#   TUNE_DIFFUSION_MODEL          True/False
#   TUNE_LLM                      True/False
#   TUNE_VISUAL                   True/False
#   TRAINING_CHECKPOINT_OUTPUT    AzureML upload target (resolved via placeholder fallback)
#   HF_TOKEN                      Required for the gated nvidia/Cosmos-Reason2-2B backbone

set -euo pipefail

echo "=== GR00T N1.7 AzureML Training ==="

# Defaults derived from NVIDIA's documented Quick-Start (1x A100 80 GB) recipe.
GR00T_REF="${GR00T_REF:-main}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_NAME="${DATASET_NAME:-schaeffler_bimanual}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/data}"
IMAGE_KEY_PRIMARY="${IMAGE_KEY_PRIMARY:-observation.images.d405_stationary_r_0}"
IMAGE_KEY_LEFT_WRIST="${IMAGE_KEY_LEFT_WRIST:-observation.images.d405_stationary_l_1}"
IMAGE_KEY_RIGHT_WRIST="${IMAGE_KEY_RIGHT_WRIST:-observation.images.d405_stationary_l_2}"
STATE_SLICES="${STATE_SLICES:-right_arm=0:6,left_arm=6:12}"
ACTION_SLICES="${ACTION_SLICES:-${STATE_SLICES}}"
ANNOTATION_MAPPING="${ANNOTATION_MAPPING:-human.task_description=task_index}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
SKIP_WEIGHT_LOADING="${SKIP_WEIGHT_LOADING:-False}"
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
TUNE_PROJECTOR="${TUNE_PROJECTOR:-True}"
TUNE_DIFFUSION_MODEL="${TUNE_DIFFUSION_MODEL:-True}"
TUNE_LLM="${TUNE_LLM:-False}"
TUNE_VISUAL="${TUNE_VISUAL:-False}"

WORKSPACE="${WORKSPACE:-/workspace}"
GR00T_DIR="${WORKSPACE}/Isaac-GR00T"

# AzureML uploads `training/` as the code asset, mounting it at the job's
# initial cwd. Capture that path before any `cd` so converter invocations and
# the embodiment-config copy can find the snapshot regardless of cwd.
CODE_DIR="$(pwd)"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# ---- 1. Resolve placeholders ----
# The AzureML K8s extension does NOT substitute ${{outputs.X}} or ${{inputs.X}}
# placeholders in environment_variables for Download / read-only-mount inputs.
# Treat any value that still contains "${{" as unsubstituted (same workaround
# the OFT pipeline uses; see commit 1ee37d45).
_strip_placeholder() {
  local v="${1:-}"
  [[ "$v" == *'${{'* ]] && echo "" || echo "$v"
}

TRAINING_CHECKPOINT_OUTPUT="$(_strip_placeholder "${TRAINING_CHECKPOINT_OUTPUT:-}")"
if [[ -z "${TRAINING_CHECKPOINT_OUTPUT}" ]] && [[ -n "${AZUREML_CR_DATA_CAPABILITY_PATH:-}" ]]; then
  TRAINING_CHECKPOINT_OUTPUT="${AZUREML_CR_DATA_CAPABILITY_PATH}/checkpoints"
fi
RUN_ROOT_DIR="${TRAINING_CHECKPOINT_OUTPUT:-${WORKSPACE}/outputs/gr00t}"

mkdir -p "${WORKSPACE}" "${DATASET_ROOT}" "${RUN_ROOT_DIR}"
echo "[output] checkpoints -> ${RUN_ROOT_DIR}"

# ---- Optional: prime output dir with a prior checkpoint for resume ----
# When the submit script wires --resume-from, the job receives a uri_folder
# input mounted read-only and surfaced as RESUME_CHECKPOINT. We symlink any
# `checkpoint-*` subdirectories into RUN_ROOT_DIR so HuggingFace Trainer's
# `get_last_checkpoint(output_dir)` (called inside
# gr00t/experiment/experiment.py via `trainer.train(resume_from_checkpoint=True)`)
# detects the most recent checkpoint and resumes from it.
#
# Notes:
#  - Symlinks (not copies) keep the resume free of bulk I/O at startup.
#  - Prior checkpoints in this repo were written with --save-only-model True,
#    so optimizer/scheduler state is absent. HF Trainer logs a warning and
#    re-initializes the optimizer; the step counter still advances from
#    trainer_state.json so max_steps and the LR schedule re-anchor to the
#    requested total.
RESUME_CHECKPOINT="$(_strip_placeholder "${RESUME_CHECKPOINT:-}")"
if [[ -n "${RESUME_CHECKPOINT}" ]] && [[ -d "${RESUME_CHECKPOINT}" ]]; then
  echo "[resume] linking checkpoints from ${RESUME_CHECKPOINT} -> ${RUN_ROOT_DIR}"
  shopt -s nullglob
  for src in "${RESUME_CHECKPOINT}"/checkpoint-*; do
    name="$(basename "${src}")"
    if [[ ! -e "${RUN_ROOT_DIR}/${name}" ]]; then
      ln -s "${src}" "${RUN_ROOT_DIR}/${name}"
      echo "[resume]   ${name}"
    fi
  done
  shopt -u nullglob
elif [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "[resume] WARNING: RESUME_CHECKPOINT=${RESUME_CHECKPOINT} not found; starting from scratch" >&2
fi

# Restore training/ prefix so `python -m training.il.scripts.gr00t.*` resolves.
if [[ ! -e training ]]; then ln -s . training; fi

# ---- 2. System deps (matches Isaac-GR00T/docker/Dockerfile) ----
echo "[system] installing OS packages"
apt-get update -qq && apt-get install -y -qq \
  build-essential git git-lfs curl wget ca-certificates \
  ffmpeg libegl1 libaio-dev \
  python3.10 python3.10-venv python3.10-dev python3-pip python-is-python3 \
  >/dev/null
git lfs install --skip-repo >/dev/null

# Install uv (Astral) into ~/.local/bin so subsequent shells can find it.
if ! command -v uv >/dev/null 2>&1; then
  echo "[system] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="${HOME}/.local/bin:${PATH}"

# ---- 3. Clone Isaac-GR00T + uv sync ----
if [[ ! -d "${GR00T_DIR}" ]]; then
  echo "[clone] Isaac-GR00T@${GR00T_REF}"
  git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "${GR00T_DIR}"
  git -C "${GR00T_DIR}" checkout "${GR00T_REF}"
  git -C "${GR00T_DIR}" submodule update --init --recursive
fi

cd "${GR00T_DIR}"

# uv sync uses Isaac-GR00T/uv.lock (pinned torch 2.7.1+cu128, transformers
# 4.57.3, flash-attn 2.7.4.post1 prebuilt wheel, deepspeed 0.17.6). The
# upstream Dockerfile passes --extra dev for interactive use; the production
# job omits it to keep the image small.
echo "[uv] sync --frozen"
uv python install 3.10 >/dev/null
uv sync --frozen --python 3.10 --no-cache

# MLflow stack for AzureML metric logging via training/il/scripts/gr00t/train.py.
# These are NOT in Isaac-GR00T's uv.lock; install into the same venv so the
# wrapper and the wandb-shim resolve a consistent mlflow import.
echo "[uv] installing mlflow + azure-ml deps"
uv pip install --quiet \
  mlflow-skinny==3.9.0 \
  azureml-mlflow==1.62.0.post2 \
  azure-ai-ml \
  azure-identity \
  psutil pynvml

# ---- 4. Resolve and convert dataset ----
DATASET_MOUNT="$(_strip_placeholder "${DATASET_MOUNT:-}")"
: "${DATASET_MOUNT:=${AZURE_ML_INPUT_dataset_asset:-}}"

if [[ -z "${DATASET_MOUNT}" ]] || [[ ! -d "${DATASET_MOUNT}" ]]; then
  # Fall back to a pre-populated DATASET_ROOT/DATASET_NAME (local dev path).
  DATASET_MOUNT="${DATASET_ROOT}/${DATASET_NAME}"
fi
if [[ ! -d "${DATASET_MOUNT}" ]]; then
  echo "ERROR: dataset not found at ${DATASET_MOUNT}" >&2
  exit 1
fi
echo "[dataset] source: ${DATASET_MOUNT}"

# AzureML uri_folder mounts are read-only; the v3 -> v2.1 converter renames
# the source folder, so copy to a writable workspace location first.
DATASET_WORK="${DATASET_ROOT}/${DATASET_NAME}"
if [[ "${DATASET_MOUNT}" != "${DATASET_WORK}" ]]; then
  echo "[dataset] copying to writable workspace at ${DATASET_WORK}"
  mkdir -p "${DATASET_ROOT}"
  rm -rf "${DATASET_WORK}"
  cp -r --reflink=auto "${DATASET_MOUNT}" "${DATASET_WORK}" 2>/dev/null \
    || cp -r "${DATASET_MOUNT}" "${DATASET_WORK}"
fi

cd "${WORKSPACE}"

# Inspect meta/info.json to choose the right converter:
#   * Per-frame Parquet (standard LeRobot v3)  -> upstream convert_v3_to_v2.py
#   * Per-frame JSONL (Schaeffler-style v3-like) -> our jsonl_to_lerobot_v21.py
src_info="${DATASET_WORK}/meta/info.json"
if [[ ! -f "${src_info}" ]]; then
  echo "ERROR: dataset missing meta/info.json at ${src_info}" >&2
  exit 1
fi
src_data_path="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('data_path',''))" "${src_info}")"

if [[ "${src_data_path}" == *.jsonl ]]; then
  echo "[convert] detected JSONL layout (${src_data_path}); running jsonl_to_lerobot_v21"
  manifest_arg=()
  if [[ -f "${DATASET_WORK}/training_manifest.json" ]]; then
    manifest_arg=( --manifest "${DATASET_WORK}/training_manifest.json" )
  fi
  uv run --project "${GR00T_DIR}" python -m training.il.scripts.gr00t.jsonl_to_lerobot_v21 \
    --source "${DATASET_WORK}" \
    --dest   "${DATASET_WORK}.v21" \
    --force \
    "${manifest_arg[@]}"
  # The converter symlinks `videos/` into the source dataset; materialize it
  # before the rename so the upcoming `mv` doesn't make videos/ point at itself.
  if [[ -L "${DATASET_WORK}.v21/videos" ]]; then
    real_videos="$(readlink -f "${DATASET_WORK}.v21/videos")"
    rm "${DATASET_WORK}.v21/videos"
    mv "${real_videos}" "${DATASET_WORK}.v21/videos"
  fi
  rm -rf "${DATASET_WORK}"
  mv "${DATASET_WORK}.v21" "${DATASET_WORK}"
else
  echo "[convert] LeRobot v3 -> v2.1 (idempotent)"
  uv run --project "${GR00T_DIR}" python -m training.il.scripts.gr00t.lerobot_v3_to_v2 \
    --gr00t-dir "${GR00T_DIR}" \
    --dataset-dir "${DATASET_WORK}" \
    --repo-id "${DATASET_NAME}"
fi

echo "[modality] writing meta/modality.json"
video_spec="front=${IMAGE_KEY_PRIMARY},wrist_left=${IMAGE_KEY_LEFT_WRIST},wrist_right=${IMAGE_KEY_RIGHT_WRIST}"
modality_args=(
  --dataset-dir "${DATASET_WORK}"
  --state-slices "${STATE_SLICES}"
  --action-slices "${ACTION_SLICES}"
  --video "${video_spec}"
  --force
)
[[ -n "${ANNOTATION_MAPPING}" ]] && modality_args+=( --annotation "${ANNOTATION_MAPPING}" )
uv run --project "${GR00T_DIR}" python -m training.il.scripts.gr00t.write_modality_json "${modality_args[@]}"

# ---- 5. Install the UR5e bimanual modality config into the GR00T tree ----
echo "[embodiment] copying ur5e_bimanual_config.py"
mkdir -p "${GR00T_DIR}/examples/UR5eBimanual"
cp "${CODE_DIR}/il/gr00t/ur5e_bimanual_config.py" \
   "${GR00T_DIR}/examples/UR5eBimanual/ur5e_bimanual_config.py"

# ---- 6. Train ----
cd "${GR00T_DIR}"

# Metrics route through Azure ML MLflow via training/il/scripts/gr00t/train.py.
# The wrapper prepends training/il/scripts/mlflow_shim/ to PYTHONPATH so the
# subprocess's `import wandb` resolves to our MLflow forwarder; GR00T's
# experiment.py is upstream-locked to HF Trainer's WandbCallback (no MLflow
# option), so this is the only way to capture its metrics without forking.
#
# We deliberately do NOT export WANDB_DISABLED/WANDB_MODE here: HF Trainer's
# integration_utils.py rejects `report_to='wandb'` + WANDB_DISABLED with a
# RuntimeError at Trainer construction, BEFORE wandb is ever imported, which
# means our shim never gets a chance to attach. The shim is the gate.

# HuggingFace cache lives under the writable workspace so weights persist
# across stages but do not pollute the read-only image layers.
export HF_HOME="${WORKSPACE}/hf-cache"
mkdir -p "${HF_HOME}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[hf] WARNING: HF_TOKEN not set; gated nvidia/Cosmos-Reason2-2B fetch will fail with 401" >&2
fi

# Boolean -> CLI flag translation. GR00T's tyro CLI accepts --tune-X / --no-tune-X
# pairs for each boolean parameter; flag presence is significant (omitting the
# flag falls back to the FinetuneConfig default).
_bool_flag() {
  local name="$1" value="$2"
  case "${value,,}" in
    true|1|yes)  echo "--${name}" ;;
    false|0|no)  echo "--no-${name}" ;;
    *) echo "ERROR: invalid bool '${value}' for ${name}" >&2; exit 2 ;;
  esac
}

# `--use-wandb` (no value) causes upstream gr00t/experiment/experiment.py to set
# HuggingFace TrainingArguments(report_to="wandb"), which registers Trainer's
# WandbCallback. Our shim under training/il/scripts/mlflow_shim/ intercepts
# every wandb.init/wandb.log call and forwards it to mlflow.
# (tyro maps Python bool fields to a presence flag pair --use-wandb /
# --no-use-wandb; passing `--use-wandb True` would be parsed as a positional.)
train_cmd=(
  uv run python -m training.il.scripts.gr00t.train
  torchrun
  --standalone --nnodes 1 --nproc-per-node "${NUM_GPUS}"
  --master-port 29500
  gr00t/experiment/launch_finetune.py
  --base-model-path "${BASE_MODEL_PATH}"
  --dataset-path "${DATASET_WORK}"
  --embodiment-tag NEW_EMBODIMENT
  --modality-config-path examples/UR5eBimanual/ur5e_bimanual_config.py
  --num-gpus "${NUM_GPUS}"
  --output-dir "${RUN_ROOT_DIR}"
  --max-steps "${MAX_STEPS}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --use-wandb
  "$(_bool_flag save-only-model     "${SAVE_ONLY_MODEL}")"
  "$(_bool_flag skip-weight-loading "${SKIP_WEIGHT_LOADING}")"
  "$(_bool_flag tune-projector      "${TUNE_PROJECTOR}")"
  "$(_bool_flag tune-diffusion-model "${TUNE_DIFFUSION_MODEL}")"
  "$(_bool_flag tune-llm            "${TUNE_LLM}")"
  "$(_bool_flag tune-visual         "${TUNE_VISUAL}")"
)

echo "[train] ${train_cmd[*]}"
exec "${train_cmd[@]}"
