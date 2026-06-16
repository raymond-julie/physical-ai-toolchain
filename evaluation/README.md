# Evaluation

Software-in-the-loop (SiL) and hardware-in-the-loop (HiL) evaluation for trained robot policies.

## 📂 Directory Structure

| Directory         | Purpose                                             |
|-------------------|-----------------------------------------------------|
| `sil/`            | SiL evaluation scripts, workflows, Docker artifacts |
| `metrics/`        | Plotting, artifact upload, MLflow bootstrapping     |
| `tests/`          | Evaluation tests                                    |
| `hil/`            | Hardware-in-the-loop evaluation (placeholder)       |
| `setup/`          | Evaluation setup scripts (placeholder)              |
| `specifications/` | Domain specifications                               |
| `examples/`       | Example configurations                              |

## 🚀 Quick Start

Submit an Isaac Lab policy evaluation:

```sh
evaluation/sil/scripts/submit-azureml-isaaclab-evaluation.sh
```

Submit a LeRobot evaluation:

```sh
evaluation/sil/scripts/submit-azureml-lerobot-eval.sh
```

## 🤖 VLA Offline Evaluators

Two Vision-Language-Action (VLA) checkpoint evaluators compute per-joint replay
error on held-out episodes of the UR5e dual-arm LeRobot dataset. They are the
VLA counterparts of the LeRobot ACT/Diffusion batch evaluator
[`sil/scripts/batch-lerobot-eval.py`](sil/scripts/batch-lerobot-eval.py): all
three load a trained policy, run inference frame-by-frame, and report MSE
against the recorded actions, but each targets a different policy family.

| Script | Policy family | Dataset | Notes |
|--------|---------------|---------|-------|
| [`sil/eval_gr00t_dual_arm.py`](sil/eval_gr00t_dual_arm.py) | GR00T-N1.5-3B (Isaac, PyTorch) | LeRobot v2.0 (GR00T flavor) | Mirrors the training data config without crop/jitter augmentation |
| [`sil/eval_openpi_ur5e_dual_arm.py`](sil/eval_openpi_ur5e_dual_arm.py) | openpi π₀ / π₀.₅ (JAX) | LeRobot v2.1 | Reuses the shared policy module from `training/vla/scripts/` |

Both require a CUDA GPU and the trained checkpoint; they are matched to the
trainers in [`training/vla/`](../training/vla/README.md).

```sh
# GR00T: report per-joint MSE over the last 7 held-out episodes.
python evaluation/sil/eval_gr00t_dual_arm.py \
    --checkpoint /outputs/.../checkpoint-100000 --dataset /data --holdout-last 7

# openpi: imports openpi_ur5e_dual_arm_policy from training/vla/scripts/ (no copy).
python evaluation/sil/eval_openpi_ur5e_dual_arm.py \
    --checkpoint /outputs/.../100000 --dataset /data --openpi-dir /opt/openpi
```

> [!NOTE]
> The openpi evaluator does not duplicate the policy/data-config code: it imports
> `openpi_ur5e_dual_arm_policy` from `training/vla/scripts/`. Override that
> location with `--vla-scripts-dir` and point `--openpi-dir` at the openpi
> checkout whose `src/` holds the `openpi` package.


> [!NOTE]
> Replace all example IP placeholders (for example, 192.168.1.x) with the actual robot IP addresses for your environment before running.
