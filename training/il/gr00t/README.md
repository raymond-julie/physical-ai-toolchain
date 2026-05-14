# GR00T N1.7 Fine-Tuning Pipeline

End-to-end pipeline to fine-tune [NVIDIA Isaac-GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) on a LeRobot dataset, submitted to Azure ML against an A100-backed compute target. Mirrors the OpenVLA-OFT pipeline structure (submit wrapper + command-job YAML + entry script + helpers) but installs GR00T at runtime in a CUDA 12.8 devel container and converts LeRobot v3 datasets to GR00T's v2.1 layout in-job.

## 🚀 Pipeline Stages

| Stage | Script | Description |
| --- | --- | --- |
| 1. Convert | [lerobot_v3_to_v2.py](../scripts/gr00t/lerobot_v3_to_v2.py) | Wrap Isaac-GR00T's `convert_v3_to_v2.py` to produce a v2.1 layout (idempotent). |
| 2. Modality | [write_modality_json.py](../scripts/gr00t/write_modality_json.py) | Author `meta/modality.json` from CLI slice specs (state/action) and camera key mappings (video). |
| 3. Embodiment | [ur5e_bimanual_config.py](ur5e_bimanual_config.py) | Register `EmbodimentTag.NEW_EMBODIMENT` with the 12-D bimanual UR5e modality config (both arms RELATIVE, 16-step horizon, no grippers). |
| 4. Train | [azureml-train-entry.sh](../scripts/gr00t/azureml-train-entry.sh) | Container entry: install GR00T via `uv sync`, convert dataset, copy embodiment config, run `torchrun gr00t/experiment/launch_finetune.py`. |
| 5. Submit | [submit-azureml-gr00t-training.sh](../scripts/submit-azureml-gr00t-training.sh) | Submission wrapper: profile selection, environment registration, `az ml job create --set` overrides. |

## 📋 Prerequisites

| Requirement | Notes |
| --- | --- |
| Azure CLI authenticated (`az login`) | Source `infrastructure/terraform/prerequisites/az-sub-init.sh` first |
| Azure ML CLI extension | `az extension add --name ml` |
| Dataset registered as an AzureML uri_folder | Use `training/il/scripts/upload-dataset-to-aml.sh` |
| HuggingFace account | Must have accepted the [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) license; `HF_TOKEN` exported |
| Point-to-site VPN connected | Required for private AKS / private workspace storage |
| A100 80 GB compute | AKS `gpu-a100` InstanceType OR managed compute `a100-cluster` |

> [!IMPORTANT]
> `nvidia/Cosmos-Reason2-2B` (the GR00T backbone) is a **gated** model. Submission without `HF_TOKEN` will fail with HTTP 401 at the first HF download unless `--skip-weight-loading True` is also passed (`smoke-a10` profile sets this automatically).

## ⚙️ Recipe (defaults)

NVIDIA's documented Quick-Start tier targeting 1× A100 80 GB. The Schaeffler bimanual UR5e (~71 k frames) sits comfortably in the documented "few thousand to 20k steps" range.

| Parameter | Value | Notes |
| --- | --- | --- |
| Container image | `nvidia/cuda:12.8.0-devel-ubuntu22.04` | Matches Isaac-GR00T/docker/Dockerfile |
| Base model | `nvidia/GR00T-N1.7-3B` | 6.93 GB BF16, not gated |
| Backbone (auto-fetched) | `nvidia/Cosmos-Reason2-2B` | Gated; `HF_TOKEN` required |
| Embodiment tag | `NEW_EMBODIMENT` | Custom UR5e via `ur5e_bimanual_config.py` |
| Action horizon | 16 steps | ~0.53 s at 30 Hz |
| Arms representation | RELATIVE | Matches launcher's hard-coded `use_relative_action=True` |
| Grippers representation | ABSOLUTE | Binary open/close |
| `num_gpus` | 1 | Single A100 80 GB |
| `global_batch_size` | 32 | Documented Quick-Start |
| `learning_rate` | 1e-4 | `FinetuneConfig` default |
| `max_steps` | 20,000 | Save every 2,000 |
| `save_total_limit` | 3 | Keep last 3 |
| `save_only_model` | True | ~7 GB/save vs ~19 GB with optimizer state |
| Tune toggles | `tune_projector=True, tune_diffusion_model=True` | LLM + visual frozen (defaults) |
| Expected peak VRAM | ~35 GB | Documented |
| Expected wall-clock | 3–5 h | Needs first-run calibration |

## 🏗️ Compute

### Option A. AKS-backed (Arc-attached AzureML extension)

Requires the `gpu-a100` InstanceType CRD. The current cluster ships only the A10 `gpu` InstanceType; bring up the A100 pool with a Terraform change before running the production profile:

```bash
# Uncomment the a100gpu pool in infrastructure/terraform/terraform.tfvars
cd infrastructure/terraform
terraform apply
kubectl apply -f ../setup/manifests/azureml-instance-types.yaml
```

### Option B. Managed compute cluster (cross-region eastus, recommended)

The westus3 workspace has only 96 vCPU A100 quota; eastus has 400 vCPU. Stand up a sibling workspace and cluster:

```bash
infrastructure/setup/setup-eastus-a100-workspace.sh
```

This creates `mlw-hex-train-eus-002` + `a100-cluster` (`Standard_NC24ads_A100_v4` = 1× A100 80GB, autoscale 0→2, 30 min idle scale-down). One A100 80 GB is the documented sufficient target for GR00T N1.7 fine-tune.

| VM size | A100 GPUs | Memory | Notes |
| --- | --- | --- | --- |
| `Standard_NC24ads_A100_v4` | 1 × 80 GB | 220 GiB | Default for managed compute; sufficient for the documented recipe |
| `Standard_NC48ads_A100_v4` | 2 × 80 GB | 440 GiB | Multi-GPU follow-on (not implemented in v1) |

## 🔧 Usage

### 0. Connect to VPN

The workspace storage account is private — uploading datasets and submitting jobs requires the point-to-site VPN.

```bash
cd infrastructure/terraform/vpn
nslookup stfyep5hexosmohack001.blob.core.windows.net  # must resolve before continuing
```

### 1. Register the dataset as an AzureML data asset

```bash
training/il/scripts/upload-dataset-to-aml.sh \
  --path datasets/schaeffler_sim_avc1/second_collection \
  --name schaeffler-sim-avc1-second \
  --version 1
```

The dataset can be either LeRobot v3 or v2.1 — the entry script's `lerobot_v3_to_v2.py` wrapper detects the codebase version and skips conversion when v2.1 is already present.

### 2. A10 smoke test (validates wrapper + install + dataset path only)

```bash
export HF_TOKEN=hf_...   # not strictly needed when --skip-weight-loading True (set by smoke-a10)
training/il/scripts/submit-azureml-gr00t-training.sh \
  --profile smoke-a10 \
  --dataset-asset schaeffler-sim-avc1-second:1 \
  --stream
```

> [!WARNING]
> A10 (24 GB) is below NVIDIA's documented 40 GB minimum for GR00T fine-tune. `launch_finetune.py` exposes no `--lora-rank`, no `--bf16`, no `--gradient-checkpointing` flag. The `smoke-a10` profile sets `--skip-weight-loading True` and trains only the diffusion head; even so, the first real training step is expected to OOM. Treat this profile as pipeline validation only — schedule `prod-a100` immediately for real training runs.

### 3. A100 production run

```bash
export HF_TOKEN=hf_...   # required — Cosmos-Reason2-2B is gated
training/il/scripts/submit-azureml-gr00t-training.sh \
  --profile prod-a100 \
  --resource-group rg-hex-train-eus-002 \
  --workspace-name mlw-hex-train-eus-002 \
  --compute a100-cluster \
  --instance-type "" \
  --dataset-asset schaeffler-sim-avc1-second:1 \
  --stream
```

> [!NOTE]
> `--instance-type ""` is correct for **managed** compute clusters. For AKS-backed compute with the `gpu-a100` InstanceType CRD, omit the flag entirely.

### 4. Override hyperparameters

All GR00T `FinetuneConfig` flags exposed as CLI arguments on the submit wrapper:

```bash
training/il/scripts/submit-azureml-gr00t-training.sh \
  --profile prod-a100 \
  --dataset-asset schaeffler-sim-avc1-second:1 \
  --global-batch-size 64 \
  --max-steps 30000 \
  --learning-rate 5e-5 \
  --tune-llm True       # unfreeze the language tower (requires bigger GPU)
```

Recipe variants:

| Variant | Flags |
| --- | --- |
| Diffusion head only | `--tune-projector False --tune-diffusion-model True --tune-llm False --tune-visual False` |
| Full backbone unlock | `--tune-projector True --tune-diffusion-model True --tune-llm True --tune-visual True` (requires >40 GB) |
| Faster save cadence | `--save-steps 500 --save-total-limit 10` |
| Resume from checkpoint | `--base-model-path /path/to/local/checkpoint --skip-weight-loading False` |

### 5. Custom embodiments

To target a different robot, copy [ur5e_bimanual_config.py](ur5e_bimanual_config.py) to a new file, adjust `modality_keys`, `delta_indices`, and `action_configs`, then submit with matching slice specs:

```bash
training/il/scripts/submit-azureml-gr00t-training.sh \
  --profile prod-a100 \
  --dataset-asset my-robot:1 \
  --state-slices "arm=0:7,gripper=7:8" \
  --action-slices "arm=0:7,gripper=7:8" \
  --image-key-primary observation.images.scene \
  --image-key-left observation.images.wrist_left \
  --image-key-right observation.images.wrist_right
```

The entry script copies the chosen embodiment file into `${GR00T_DIR}/examples/UR5eBimanual/` inside the container; to swap embodiment files, replace `ur5e_bimanual_config.py` or extend the wrapper to accept `--modality-config-source`.

## 🔍 Divergence from OpenVLA-OFT

| Aspect | OpenVLA-OFT | GR00T N1.7 |
| --- | --- | --- |
| Base image | `pytorch/pytorch:2.2.0-cuda12.1` | `nvidia/cuda:12.8.0-devel-ubuntu22.04` |
| Python | 3.10 via uv | 3.10 via uv (locked at == 3.10.\*) |
| Install | `pip install -e` repos + flash-attn 2.5.5 | `uv sync --frozen` (torch 2.7.1+cu128, flash-attn 2.7.4.post1 wheel) |
| Dataset format | LeRobot v3 → RLDS / TFDS | LeRobot v3 → LeRobot v2.1 + `meta/modality.json` |
| Embodiment | OXE `configs/transforms/mixtures` patches | `register_modality_config(EmbodimentTag.NEW_EMBODIMENT)` |
| Action representation | absolute (12-D) | both arms RELATIVE (12-D) |
| PEFT | `--lora-rank 32` (LoRA on Llama) | none (no `--lora-rank` flag; full FP32 master weights) |
| Backbone | `openvla/openvla-7b` (open) | `nvidia/Cosmos-Reason2-2B` (gated) |
| Single A100 sufficient | yes (~73 GB) | yes (~35 GB) |
| Two-GPU sweet spot | `--num-gpus 2 --batch-size 4` (default) | deferred to v2 |

## 🗑️ Cleanup

Checkpoints land under the run's `outputs/checkpoints/` and incur blob storage cost. After validating an experiment, prune via the AzureML portal or `az ml job archive --name <job>`.

## 🔍 Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `401 Unauthorized` fetching Cosmos-Reason2-2B | `HF_TOKEN` not plumbed through; pass `--hf-token` or export `HF_TOKEN` before submission |
| `OOM` at first step on A10 | Expected. A10 24 GB cannot host GR00T N1.7 fine-tune; switch to A100 |
| `Dataset not found at /workspace/data/...` | `--dataset-asset` not specified or the asset version doesn't exist; check `az ml data list` |
| `flash_attn` wheel download fails | Network policy on the AKS pool; ensure egress to `https://pypi.nvidia.com` and `https://download.pytorch.org` |
| Checkpoints saved but not uploaded | `TRAINING_CHECKPOINT_OUTPUT` placeholder didn't substitute; the entry script's fallback to `${AZUREML_CR_DATA_CAPABILITY_PATH}/checkpoints` covers this — verify in the job's user_logs |
| `RuntimeError: NVIDIA driver too old` | Node's NVIDIA driver is older than CUDA 12.8 requires (560.x+); rebuild the GPU pool or use a node with newer driver |

## 📚 Related

* [training/il/openvla_oft](../openvla_oft) — sibling OpenVLA-OFT pipeline this integration mirrors
* [training/il/lerobot](../lerobot) — LeRobot ACT/Diffusion training (no VLA)
* [.copilot-tracking/research/2026-05-13/gr00t-n17-toolchain-integration-research.md](../../../.copilot-tracking/research/2026-05-13/gr00t-n17-toolchain-integration-research.md) — research document underlying this integration
* [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — upstream GR00T repo (Apache 2.0 code; NVIDIA Open Model License weights)
* [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) — base model card
