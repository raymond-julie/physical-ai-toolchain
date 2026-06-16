# VLA Training

Vision-Language-Action (VLA) training for transformer policies that map camera
views, robot state, and a natural-language task prompt to robot actions. Two
model families are supported side by side on the same LeRobot datasets:

- **NVIDIA Isaac GR00T-N1.5-3B** — a PyTorch / HuggingFace `Trainer` fine-tune.
- **openpi π₀ / π₀.₅** (Physical Intelligence) — a JAX fine-tune with custom
  LeRobot data configs.

Both target two UR embodiments and run as OSMO workflows (or, for GR00T, on a
standalone Azure H100 VM). Offline evaluation lives in the evaluation domain
(see the Evaluation section below).

## 📁 Directory Structure

```text
vla/
├── scripts/                              # Trainers, openpi policies, OSMO submission
│   ├── train_gr00t_dual_arm.py           # GR00T UR5e dual-arm (14-DoF, 4 cam, v2.0)
│   ├── train_gr00t_dual_arm_n1_5_3b.py   # GR00T UR5e dual-arm (v2.1, mixture-capable)
│   ├── train_gr00t_ur10e.py              # GR00T UR10e single-arm (7-DoF, 2 cam)
│   ├── train_openpi_ur5e_dual_arm.py     # openpi UR5e dual-arm driver
│   ├── train_openpi_ur10e.py             # openpi UR10e single-arm driver
│   ├── openpi_ur5e_dual_arm_policy.py    # openpi dual-arm policy + data config
│   ├── openpi_ur10e_policy.py            # openpi single-arm policy + data config
│   ├── submit-osmo-gr00t-training.sh     # Render + submit a GR00T OSMO workflow
│   ├── submit-osmo-openpi-training.sh    # Render + submit an openpi OSMO workflow
│   └── h100/                             # Standalone Azure H100 VM track (reserved)
├── workflows/
│   └── osmo/                             # OSMO workflow templates + pod templates
├── tests/                                # Behavior tests (no GPU)
├── .amlignore                            # AzureML code snapshot exclusions
└── README.md                            # This file
```

## 🧠 Model Tracks

| Aspect          | GR00T-N1.5-3B                                     | openpi π₀ / π₀.₅                                  |
|-----------------|--------------------------------------------------|--------------------------------------------------|
| Framework       | PyTorch, HuggingFace `Trainer`                   | JAX / XLA, FSDP, LoRA or full fine-tune          |
| Install         | `isaac-gr00t @ n1.5-release` + transformers      | openpi checkout + JAX-CUDA + pinned LeRobot      |
| Entry points    | `train_gr00t_*.py`                               | `train_openpi_*.py` + `openpi_*_policy.py`       |
| Dataset format  | LeRobot v2.x (GR00T flavor: `modality.json`, `state.*` / `action.*` keys) | LeRobot v2.1 (flat `observation.images.color_*`, 14/7-DoF state/action) |
| Checkpoints     | `checkpoint-{step}/` + `trainer_state.json`      | `checkpoints/<config>/<exp>/<step>/`             |
| Optimizer       | `adamw_bnb_8bit` / `adamw_torch`, bf16           | LoRA or full fine-tune via openpi `TrainConfig`  |

> [!NOTE]
> The openpi trainers import the matching `openpi_*_policy.py` module and
> register a `TrainConfig` with openpi at runtime; the policy modules are the
> single source of truth for the dual-arm and single-arm data layouts and are
> reused by the offline evaluators.

## 🦾 Embodiments

| Embodiment      | DoF | Cameras | Trainers                                                          |
|-----------------|-----|---------|-------------------------------------------------------------------|
| UR5e dual-arm   | 14  | 4       | `train_gr00t_dual_arm.py`, `train_gr00t_dual_arm_n1_5_3b.py`, `train_openpi_ur5e_dual_arm.py` |
| UR10e single-arm| 7   | 2       | `train_gr00t_ur10e.py`, `train_openpi_ur10e.py`                   |

The dual-arm state and action vectors are laid out as
`[r1_joint0..5, r1_gripper, r2_joint0..5, r2_gripper]`. The openpi dual-arm
policy maps the four cameras onto base / left-wrist / right-wrist views and
drops the secondary third-person view (`color_1`) by default.

## 🚀 OSMO Submission

Each model family has a submission script that inlines the selected trainer (and,
for openpi, the policy module) into an OSMO workflow template, then submits it.
Both source `scripts/lib/common.sh`, accept an `--embodiment` flag, and support
`--config-preview`.

| Script                                                            | Workflow template                              |
|-------------------------------------------------------------------|------------------------------------------------|
| [`scripts/submit-osmo-gr00t-training.sh`](scripts/submit-osmo-gr00t-training.sh)   | [`workflows/osmo/gr00t-train.yaml`](workflows/osmo/gr00t-train.yaml)   |
| [`scripts/submit-osmo-openpi-training.sh`](scripts/submit-osmo-openpi-training.sh) | [`workflows/osmo/openpi-train.yaml`](workflows/osmo/openpi-train.yaml) |

```sh
# Submit a GR00T dual-arm fine-tune (preview the rendered config first).
training/vla/scripts/submit-osmo-gr00t-training.sh --embodiment dual_arm --config-preview
```

### Dataset source variants

The OSMO templates come in two dataset-mount flavors:

| Variant | Workflows                                                              | Dataset source                          |
|---------|-----------------------------------------------------------------------|-----------------------------------------|
| S3      | `gr00t-train.yaml`, `openpi-train.yaml`                                | `aws s3 sync` from in-cluster LocalStack S3 |
| NFS     | `train-lerobot.yaml`, `train-lerobot-cloud.yaml`                       | NFS-mounted dataset (`/data` read-only) |

The NFS variant pairs with the OSMO pod templates under
[`workflows/osmo/pod-templates/`](workflows/osmo/pod-templates/)
(`lerobot-nfs`, `gpu-5090`, `gpu-h100`, `azureml-creds`). The
`train-lerobot-cloud.yaml` variant additionally wires AzureML MLflow tracking.

## 🖥️ H100 Standalone Track

`scripts/h100/` is the home for the standalone training backend: GR00T runs
inside the same NGC container on a single Azure H100 Spot VM via Docker, with no
OSMO control plane, S3, or NFS (datasets and outputs are local bind mounts).

> [!NOTE]
> This directory is reserved and not yet populated. The H100 helper scripts
> (VM start, dataset upload, containerized train) are tracked as follow-on work.

## 📊 Evaluation

Offline checkpoint evaluation lives in the evaluation domain. Both evaluators
replay held-out episodes and report per-joint MSE:

| Evaluator                                                                          | Pairs with                                  |
|------------------------------------------------------------------------------------|---------------------------------------------|
| [`evaluation/sil/eval_gr00t_dual_arm.py`](../../evaluation/sil/eval_gr00t_dual_arm.py)             | `train_gr00t_dual_arm.py`                   |
| [`evaluation/sil/eval_openpi_ur5e_dual_arm.py`](../../evaluation/sil/eval_openpi_ur5e_dual_arm.py) | `train_openpi_ur5e_dual_arm.py`             |

The openpi evaluator imports `openpi_ur5e_dual_arm_policy` directly from
`scripts/` (it is not copied into the evaluation domain). See the
[Evaluation README](../../evaluation/README.md#-vla-offline-evaluators).

## 🔁 Azure ML Mirror

> [!NOTE]
> The reconciled AzureML replay/mirror job is the top-level
> [`workflows/osmo/replay-azureml.yaml`](../../workflows/osmo/replay-azureml.yaml)
> (with `requirements-aml-mirror.txt`), not a copy under `vla/`. Use it to mirror
> completed OSMO runs to Azure ML for model versioning rather than adding a
> per-domain duplicate.

## 📋 Specifications

See the [VLA Training Specification](../specifications/vla-training.specification.md)
for the approach, components, and dataset contracts.


> [!NOTE]
> Replace all example IP placeholders (for example, 192.168.1.x) with the actual robot IP addresses for your environment before running.
