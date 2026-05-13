# OpenVLA-OFT Fine-Tuning Pipeline

End-to-end pipeline to fine-tune [OpenVLA](https://github.com/openvla/openvla) with the [Optimized Fine-Tuning (OFT) recipe](https://openvla-oft.github.io/) on a LeRobot v3 dataset, submitted to Azure ML against an A100-backed compute target.

## Pipeline Stages

| Stage | Script | Description |
| --- | --- | --- |
| 1. Filter | [filter_dataset.py](../scripts/openvla_oft/filter_dataset.py) | Walk LeRobot v3 metadata, keep episodes that have all required video views (and optionally were judged successful by `evaluation/vlm_judge/run.py`). Emits a manifest JSON. |
| 2. Convert | [lerobot_to_rlds.py](../scripts/openvla_oft/lerobot_to_rlds.py) | Build an RLDS / TFDS dataset from the manifest (3 image views + 12-DOF proprio + 12-DOF action + per-episode language instruction). |
| 3. Register | [dataset_registration.py](../scripts/openvla_oft/dataset_registration.py) | Patch the `prismatic/vla/datasets/rlds/oxe/{configs,transforms,mixtures}.py` and `prismatic/vla/constants.py` files inside a clone of `moojink/openvla-oft` so the OFT data loader recognises our dataset. |
| 4. Train | [azureml-train-entry.sh](../scripts/openvla_oft/azureml-train-entry.sh) | Container entrypoint that runs stages 1-3 then launches `torchrun vla-scripts/finetune.py` with our hyperparameters. |
| 5. Submit | [submit-azureml-openvla-oft-training.sh](../scripts/submit-azureml-openvla-oft-training.sh) | Submission wrapper (mirrors `submit-azureml-lerobot-training.sh`) that registers the environment, builds the `--set` overrides, and invokes `az ml job create`. |

## Recipe (defaults)

OFT+ for ALOHA-style bimanual setups, adapted to the Schaeffler 12-DOF UR5e:

| Parameter | Value | Notes |
| --- | --- | --- |
| Base VLA | `openvla/openvla-7b` | 7B Llama-2 + Prismatic vision backbone |
| Images in input | 3 | primary (`d405_stationary_r_0`) + left wrist (`l_1`) + right wrist (`l_2`) |
| Proprio | enabled | 12-DOF joint state (R/L 1..6) |
| FiLM | enabled | needed for 11-sub-task language grounding |
| L1 regression head | enabled | OFT default; outperforms diffusion for our data shape |
| Action chunk | 25 | ~0.83 s at 30 Hz |
| LoRA rank | 32 | as in paper |
| Batch size / GPU | 4 | ~73 GB recommended footprint per A100 |
| Learning rate | 5e-4 | decay 10x after 50k steps |
| Max steps | 100,005 | save every 10k |
| GPUs | 2 | `Standard_NC48ads_A100_v4` (2x A100 80GB) |

## Compute

The pipeline assumes an Azure ML compute target attached to an AKS cluster running the AzureML extension (the canonical pattern in `infrastructure/`).

| Asset | Location | Purpose |
| --- | --- | --- |
| `gpu-a100` InstanceType | [azureml-instance-types.yaml](../../../infrastructure/setup/manifests/azureml-instance-types.yaml) | Requests 2x `nvidia.com/gpu`, 220 GiB RAM; nodeSelector `accelerator: nvidia` + `gpu-class: a100`. |
| `a100gpu` node pool example | [terraform.tfvars.example](../../../infrastructure/terraform/terraform.tfvars.example) | Adds an `a100gpu` pool with `Standard_NC48ads_A100_v4` and the `gpu-class: a100` label. |

> [!IMPORTANT]
> The cluster has **no A100 pool** by default. To run this pipeline, uncomment the `a100gpu` block in `terraform.tfvars`, `terraform apply` from `infrastructure/terraform/`, then re-apply the instance-type manifest via `kubectl apply -f infrastructure/setup/manifests/azureml-instance-types.yaml`.

Alternative VM sizes that satisfy the `gpu-a100` InstanceType (need 2 GPUs visible on a single node, ≥220 GiB RAM):

| VM size | A100 GPUs | Memory | Notes |
| --- | --- | --- | --- |
| `Standard_NC48ads_A100_v4` | 2 × 80 GB | 440 GiB | Default; cheapest 2-GPU A100 node |
| `Standard_NC96ads_A100_v4` | 4 × 80 GB | 880 GiB | Drop `num_gpus=4` in the submit flags to use all 4 |
| `Standard_ND96amsr_A100_v4` | 8 × 80 GB | 1900 GiB | Multi-node OFT (paper config); set `num_gpus=8` |

## Usage

### 1. Filter the dataset (local)

```bash
python -m training.il.scripts.openvla_oft.filter_dataset \
  --dataset datasets/schaeffler_sim_avc1/second_collection \
  --image-keys observation.images.d405_stationary_r_0 \
               observation.images.d405_stationary_l_1 \
               observation.images.d405_stationary_l_2 \
  --vlm-judge outputs/dataset-analysis/schaeffler_second_collection/vlm-judge.jsonl \
  --require-vlm-success \
  --output datasets/schaeffler_sim_avc1/second_collection/training_manifest.json
```

Result for `schaeffler_sim_avc1/second_collection`: **76 eligible episodes / 68,089 frames** (97 declared - 16 missing-views - 4 VLM-judged failures - 1 unjudged).

### 2. Dry-run the RLDS converter (local)

```bash
python -m training.il.scripts.openvla_oft.lerobot_to_rlds \
  --manifest datasets/schaeffler_sim_avc1/second_collection/training_manifest.json \
  --primary-camera observation.images.d405_stationary_r_0 \
  --left-wrist    observation.images.d405_stationary_l_1 \
  --right-wrist   observation.images.d405_stationary_l_2 \
  --dry-run
```

This validates the manifest + every video file is on disk and decodable. A full local build requires `tensorflow` + `tensorflow_datasets` + `decord` and writes ~30-60 GB.

### 3. Submit the AzureML job

```bash
# Preview the resolved configuration (no submission)
training/il/scripts/submit-azureml-openvla-oft-training.sh \
  -d schaeffler_sim_avc1/second_collection \
  --config-preview

# Actual submission (requires `az login` and the gpu-a100 InstanceType deployed)
training/il/scripts/submit-azureml-openvla-oft-training.sh \
  -d schaeffler_sim_avc1/second_collection \
  --blob-url "https://<account>.blob.core.windows.net/datasets/schaeffler_sim_avc1/second_collection" \
  --instance-type gpu-a100 \
  --num-gpus 2 \
  --stream
```

The blob URL must point at a folder mirroring the LeRobot v3 layout (`meta/`, `data/`, `videos/`). The container mounts this read-only at `$DATASET_ROOT/$DATASET_REPO_ID` and the entry script rebuilds the RLDS dataset in-container so it survives node-local scratch only.

### 4. Override hyperparameters

All OFT flags exposed as CLI arguments on `submit-azureml-openvla-oft-training.sh`:

```bash
training/il/scripts/submit-azureml-openvla-oft-training.sh \
  -d schaeffler_sim_avc1/second_collection \
  --batch-size 8 \
  --max-steps 150005 \
  --num-steps-before-decay 100000 \
  --lora-rank 64 \
  --use-film False        # drop FiLM if language doesn't change
```

Recipe overrides (recipe ablations):

| Variant | Flags |
| --- | --- |
| OFT (no FiLM) | `--use-film False` |
| OFT 2-image (LIBERO-style) | `--num-images 2 --num-actions-chunk 8` |
| Single-GPU dev run | `--num-gpus 1 --batch-size 1 --max-steps 1000` |

## Memory-aware Defaults

| GPU | Recommended config |
| --- | --- |
| 1 × A100 40GB | `--batch-size 1 --num-gpus 1` (75 GB recommended -> use grad accumulation) |
| 1 × A100 80GB | `--batch-size 4 --num-gpus 1` |
| 2 × A100 80GB | `--batch-size 4 --num-gpus 2` (default) |
| 4-8 × A100 80GB | `--batch-size 8 --num-gpus 4` (paper-style) |

> [!NOTE]
> OFT recommends merging the LoRA adapter into the base VLA **on the same GPU class used for inference**. If you train on A100 and deploy on H100, use `vla-scripts/merge_lora_weights_and_save.py` on the inference target. The entry script writes the LoRA adapter alongside the merged checkpoint under `$TRAINING_CHECKPOINT_OUTPUT/`.

## Related

- [evaluation/vlm_judge](../../../evaluation/vlm_judge) - generates the success labels consumed by `--require-vlm-success`
- [training/il/lerobot](../lerobot) - the ACT / Diffusion training pipeline (LeRobot, not OpenVLA)
- [docs/contributing/architecture.md](../../../docs/contributing/architecture.md) - overall toolchain architecture
