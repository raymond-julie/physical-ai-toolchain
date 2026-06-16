---
title: Operating VLA Training and Evaluation
description: End-to-end tutorial for training and evaluating the new GR00T and openpi VLA projects on UR5e and UR10e datasets
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: tutorial
keywords:
  - vla
  - gr00t
  - openpi
  - training
  - evaluation
  - osmo
estimated_reading_time: 28
sidebar_position: 4
---

Use the new VLA training projects as a structured pipeline: prepare the right dataset shape, pick the correct model family, submit the matching workflow, and validate the resulting checkpoint with offline evaluation before you promote it to inference.

## 📋 Prerequisites

| Requirement | Details |
| --- | --- |
| Dataset | LeRobot dataset prepared for the selected embodiment |
| OSMO | Running control plane and backend when using workflow submission |
| GPU capacity | Enough GPU for the chosen model family |
| Environment | Python 3.10 or later, plus the model-specific dependencies |
| Paths and IPs | Replace any example placeholders such as `192.168.1.x` with the real addresses used by your storage or robot environment |

## 🧠 Understand the Two Training Tracks

| Track | Use it when | Main entrypoints |
| --- | --- | --- |
| GR00T N1.5 | You want NVIDIA's VLA stack and the deployed inference server expects a GR00T checkpoint | `train_gr00t_dual_arm.py`, `train_gr00t_dual_arm_n1_5_3b.py`, `train_gr00t_ur10e.py` |
| openpi π₀ / π₀.₅ | You want the openpi training flow and the shared openpi policy/data-config modules | `train_openpi_ur5e_dual_arm.py`, `train_openpi_ur10e.py`, `openpi_*_policy.py` |

## 🚀 Step 1: Pick the Embodiment First

The project supports two hardware layouts:

| Embodiment | DoF | Cameras | Typical scripts |
| --- | --- | --- | --- |
| UR5e dual-arm | 14 | 4 | `train_gr00t_dual_arm.py`, `train_openpi_ur5e_dual_arm.py` |
| UR10e single-arm | 7 | 2 | `train_gr00t_ur10e.py`, `train_openpi_ur10e.py` |

Do not start with the model family. Start with the embodiment, because the dataset shape, camera layout, and state/action semantics all depend on it.

## 🗃️ Step 2: Validate the Dataset Contract

Before you submit anything, confirm the dataset matches the intended consumer:

### GR00T dataset expectations

Use GR00T when your dataset looks like this:

1. GR00T-flavored LeRobot layout.
2. `modality.json` present.
3. State and action keys split into modality groups such as `state.robot1_arm`.

### openpi dataset expectations

Use openpi when your dataset looks like this:

1. LeRobot v2.1 flat key layout.
2. Image keys such as `observation.images.color_*`.
3. Flat 14-DoF or 7-DoF state and action vectors.

If your capture output is not already in the needed format, normalize it before training with the tools in `data-management/tools`.

## 🧪 Step 3: Run a Submission Preview

Always render the workflow first.

### GR00T preview

```bash
training/vla/scripts/submit-osmo-gr00t-training.sh --embodiment dual_arm --config-preview
```

### openpi preview

```bash
training/vla/scripts/submit-osmo-openpi-training.sh --embodiment dual_arm --config-preview
```

Review:

1. Dataset path.
2. Embodiment selection.
3. Trainer entrypoint.
4. GPU pod template.
5. Checkpoint output path.

## 🧬 Step 4: Submit the Training Job

### GR00T dual-arm example

```bash
training/vla/scripts/submit-osmo-gr00t-training.sh --embodiment dual_arm
```

### openpi dual-arm example

```bash
training/vla/scripts/submit-osmo-openpi-training.sh --embodiment dual_arm
```

The submit script handles template rendering and sends the resolved workflow to OSMO.

## 🧱 Step 5: Understand the Workflow Files You Are Actually Running

| File | Role |
| --- | --- |
| `workflows/osmo/gr00t-train.yaml` | GR00T training workflow template |
| `workflows/osmo/openpi-train.yaml` | openpi training workflow template |
| `workflows/osmo/train-lerobot.yaml` | NFS-mounted dataset variant |
| `workflows/osmo/train-lerobot-cloud.yaml` | NFS + Azure ML mirror variant |
| `workflows/osmo/pod-templates/*.json` | GPU and storage execution environments |

When a run behaves incorrectly, inspect the rendered workflow first. Most problems in this stack come from the data mount, pod template, or trainer selection, not from the core training script.

## 📊 Step 6: Monitor the Run

Monitor the workflow at three levels:

1. OSMO workflow status.
2. Pod logs.
3. Checkpoint output directory.

Use the same monitoring pattern for both model families:

```bash
kubectl get pods -n osmo-workflows --watch
```

If the job fails quickly, inspect:

1. Dataset mount path.
2. Missing training dependency.
3. GPU allocation or node selection.
4. Policy/data-config mismatch.

## 📉 Step 7: Evaluate the Result Before Deployment

### Evaluate a GR00T checkpoint

```bash
python evaluation/sil/eval_gr00t_dual_arm.py \
  --checkpoint /outputs/.../checkpoint-100000 \
  --dataset /data \
  --holdout-last 7
```

### Evaluate an openpi checkpoint

```bash
python evaluation/sil/eval_openpi_ur5e_dual_arm.py \
  --checkpoint /outputs/.../100000 \
  --dataset /data \
  --openpi-dir /opt/openpi \
  --holdout-last 7
```

These offline evaluators replay held-out frames and compute per-joint MSE. Use them to reject clearly broken checkpoints before anyone wires them into an edge inference deployment.

## 🔁 Step 8: Decide What Happens After Training

After you have a passing checkpoint, choose one of these paths:

| Next step | Use it when |
| --- | --- |
| Azure ML mirror | You want model registration and experiment tracking |
| GR00T edge deployment | You trained a GR00T checkpoint and want it served by the deployed N1.5 inference stack |
| Further offline comparison | You want to compare multiple checkpoints by evaluation results before promotion |

The Azure ML mirror flow lives at the repository-level `workflows/osmo/replay-azureml.yaml`, not inside `training/vla`.

## ✅ Verification Checklist

You have a healthy VLA flow when:

1. The workflow renders correctly in preview mode.
2. The selected trainer matches the embodiment.
3. The job completes and emits checkpoints.
4. Offline evaluation produces usable MSE output.
5. The promoted checkpoint matches the intended downstream consumer (GR00T or openpi).

## 🔗 Related Documentation

- [Your First LeRobot Training Job](your-first-lerobot-training-job.md)
- [End-to-End LeRobot Pipeline](end-to-end-lerobot-pipeline.md)
- [Evaluation README](../../evaluation/README.md)
