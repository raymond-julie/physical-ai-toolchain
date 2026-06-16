#!/usr/bin/env python3
"""Offline evaluation of an openpi (π₀ / π₀.₅) checkpoint on held-out episodes.

Loads a trained openpi checkpoint, runs inference on held-out episodes from the
UR5e dual-arm LeRobot dataset, and reports per-joint MSE between predicted and
ground-truth actions. This is the openpi (Physical Intelligence, JAX)
counterpart to ``scripts/batch-lerobot-eval.py`` (LeRobot ACT/Diffusion) and to
``eval_gr00t_dual_arm.py`` (GR00T); all three compute offline replay error, but
this script reuses the shared openpi policy/data-config module rather than
redefining it.

The policy/data-config module (``openpi_ur5e_dual_arm_policy``) is NOT
duplicated here: it is imported from ``training/vla/scripts/`` (override the
location with ``--vla-scripts-dir``). The openpi source tree itself is supplied
via ``--openpi-dir`` (default ``/opt/openpi``).

Usage:
    uv run eval_openpi_ur5e_dual_arm.py \\
        --checkpoint /outputs/openpi-ur5e-dual/checkpoints/pi05_ur5e_dual_lora/ur5e_dual_run/100000 \\
        --dataset /data --holdout-last 7

    # Or specify explicit episodes:
    uv run eval_openpi_ur5e_dual_arm.py \\
        --checkpoint /outputs/.../100000 \\
        --dataset /data --holdout-episodes 50,51,52,53,54,55,56
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATASET = Path(os.environ.get("DATASET_PATH", "/data"))
DEFAULT_OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/outputs/openpi-ur5e-dual"))
DEFAULT_OPENPI_DIR = Path(os.environ.get("OPENPI_DIR", "/opt/openpi"))
# The shared openpi policy/data-config module lives with the training scripts.
DEFAULT_VLA_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "training" / "vla" / "scripts"

# 14-DoF dual-arm action layout: [r1_j0..5, r1_grip, r2_j0..5, r2_grip].
DIM_LABELS = [
    "R1_j0", "R1_j1", "R1_j2", "R1_j3", "R1_j4", "R1_j5", "R1_grip",
    "R2_j0", "R2_j1", "R2_j2", "R2_j3", "R2_j4", "R2_j5", "R2_grip",
]

_LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to checkpoint dir. Auto-detects latest if omitted.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Base output dir (for auto-detecting checkpoints).")
    parser.add_argument("--openpi-dir", type=Path, default=DEFAULT_OPENPI_DIR,
                        help="Path to the openpi checkout (its src/ is added to sys.path).")
    parser.add_argument("--vla-scripts-dir", type=Path, default=DEFAULT_VLA_SCRIPTS_DIR,
                        help="Directory containing openpi_ur5e_dual_arm_policy.py "
                             "(default: training/vla/scripts).")
    parser.add_argument("--holdout-episodes", type=str, default=None,
                        help="Comma-separated episode indices to evaluate on.")
    parser.add_argument("--holdout-last", type=int, default=7,
                        help="Hold out the last N episodes (default: 7).")
    parser.add_argument("--stride", type=int, default=10,
                        help="Evaluate every Nth frame per episode (default: 10).")
    parser.add_argument("--pi05", action="store_true", default=True)
    parser.add_argument("--pi0", dest="pi05", action="store_false")
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--use-secondary-base", action="store_true", default=False)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--results-file", type=Path, default=None,
                        help="Save results JSON to this path.")
    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    """Return the checkpoint path, auto-detecting the latest when unspecified."""
    if args.checkpoint:
        return args.checkpoint

    if args.pi05:
        cfg_name = "pi05_ur5e_dual_lora" if args.lora else "pi05_ur5e_dual"
    else:
        cfg_name = "pi0_ur5e_dual_lora" if args.lora else "pi0_ur5e_dual"

    ckpt_base = args.output / "checkpoints" / cfg_name
    if not ckpt_base.exists():
        _LOGGER.error("Checkpoint base dir not found: %s", ckpt_base)
        sys.exit(1)

    exp_dirs = sorted(ckpt_base.iterdir())
    if not exp_dirs:
        _LOGGER.error("No experiment dirs in %s", ckpt_base)
        sys.exit(1)

    numbered = sorted(
        (d for d in exp_dirs[-1].iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not numbered:
        _LOGGER.error("No numbered checkpoints in %s", exp_dirs[-1])
        sys.exit(1)

    latest = numbered[-1]
    _LOGGER.info("Auto-detected latest checkpoint: %s", latest)
    return latest


def _get_holdout_episodes(args: argparse.Namespace) -> list[int]:
    """Determine which episodes to hold out for evaluation."""
    if args.holdout_episodes:
        return [int(x) for x in args.holdout_episodes.split(",")]

    info_path = args.dataset / "meta" / "info.json"
    total_episodes = 57
    if info_path.exists():
        info = json.loads(info_path.read_text())
        total_episodes = info.get("total_episodes", total_episodes)

    return list(range(total_episodes - args.holdout_last, total_episodes))


def _evaluate(
    policy: Any,
    dataset: Any,
    holdout_indices: list[int],
    stride: int,
) -> dict[str, Any]:
    """Run evaluation on held-out frames and compute per-joint MSE."""
    all_errors: list[np.ndarray] = []
    per_episode: dict[int, dict[str, Any]] = {}

    frame_episodes: dict[int, list[int]] = {}
    for i in range(len(dataset)):
        ep_idx = dataset[i].get("episode_index", None)
        if ep_idx is not None:
            frame_episodes.setdefault(ep_idx, []).append(i)

    for ep_idx in holdout_indices:
        if ep_idx not in frame_episodes:
            _LOGGER.warning("Episode %d not found in dataset, skipping", ep_idx)
            continue

        ep_errors: list[np.ndarray] = []
        for frame_i in frame_episodes[ep_idx][::stride]:
            row = dataset[frame_i]
            gt_action = row.get("action", None)
            if gt_action is None:
                continue
            gt_flat = np.asarray(gt_action).flatten()

            observation = {k: v for k, v in row.items() if k.startswith("observation.") or k == "prompt"}
            predicted = policy.infer(observation)
            pred_action = np.asarray(predicted.get("actions", predicted.get("action", [])))
            pred_flat = pred_action[0].flatten() if pred_action.ndim == 2 else pred_action.flatten()

            min_len = min(len(gt_flat), len(pred_flat))
            if min_len == 0:
                continue
            ep_errors.append((gt_flat[:min_len] - pred_flat[:min_len]) ** 2)

        if ep_errors:
            ep_mse = float(np.mean(ep_errors))
            per_episode[ep_idx] = {
                "mse": ep_mse,
                "frames": len(ep_errors),
                "per_dim_mse": np.mean(ep_errors, axis=0).tolist(),
            }
            all_errors.extend(ep_errors)
            _LOGGER.info("Episode %d: MSE=%.6f (%d frames)", ep_idx, ep_mse, len(ep_errors))

    if not all_errors:
        return {"error": "No frames evaluated"}

    per_dim = np.mean(all_errors, axis=0).tolist()
    return {
        "overall_mse": float(np.mean(all_errors)),
        "per_dim_mse": per_dim,
        "dim_labels": DIM_LABELS[: len(per_dim)],
        "total_frames": sum(e["frames"] for e in per_episode.values()),
        "episodes_evaluated": len(per_episode),
        "per_episode": per_episode,
    }


def _print_summary(checkpoint_path: Path, results: dict[str, Any]) -> None:
    overall = results.get("overall_mse")
    overall_str = f"{overall:.6f}" if overall is not None else "N/A"

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (openpi π₀.₅)")
    print("=" * 60)
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Episodes evaluated: {results.get('episodes_evaluated', 0)}")
    print(f"Total frames:       {results.get('total_frames', 0)}")
    print(f"Overall MSE:        {overall_str}")
    print("=" * 60)

    if "per_dim_mse" in results and "dim_labels" in results:
        print("\nPer-dimension MSE:")
        for label, mse in zip(results["dim_labels"], results["per_dim_mse"], strict=False):
            print(f"  {label:8s}: {mse:.6f}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = create_parser().parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.openpi_dir = args.openpi_dir.resolve()

    # Make the shared openpi policy/data-config module and the openpi source tree
    # importable. The policy module is NOT vendored here; it ships with the
    # training scripts under training/vla/scripts/.
    vla_scripts = args.vla_scripts_dir.resolve()
    if vla_scripts.is_dir():
        sys.path.insert(0, str(vla_scripts))
    openpi_src = args.openpi_dir / "src"
    if openpi_src.is_dir():
        sys.path.insert(0, str(openpi_src))

    try:
        import openpi_ur5e_dual_arm_policy as policy_mod
    except ImportError:
        _LOGGER.exception("Failed to import openpi_ur5e_dual_arm_policy")
        _LOGGER.error("Ensure --vla-scripts-dir (%s) and --openpi-dir (%s) are valid", vla_scripts, args.openpi_dir)
        return 1

    checkpoint_path = _resolve_checkpoint(args)
    holdout = _get_holdout_episodes(args)
    _LOGGER.info("Checkpoint: %s", checkpoint_path)
    _LOGGER.info("Holdout episodes: %s", holdout)

    repo_id = str(args.dataset)
    config = policy_mod.build_train_configs(
        repo_id=repo_id,
        exp_name="eval",
        num_train_steps=1,
        batch_size=1,
        save_interval=1,
        lora=args.lora,
        pi05=args.pi05,
        default_prompt=args.default_prompt,
        use_secondary_base=args.use_secondary_base,
        assets_base_dir=str(args.output / "assets"),
        checkpoint_base_dir=str(args.output / "checkpoints"),
    )
    policy_mod.register(config)

    from openpi.policies import policy_loader

    policy = policy_loader.PolicyLoader(
        config=config,
        checkpoint_path=str(checkpoint_path),
    ).load()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, local_files_only=True)
    _LOGGER.info("Dataset loaded: %d frames", len(dataset))

    results = _evaluate(policy, dataset, holdout, args.stride)
    _print_summary(checkpoint_path, results)

    if args.results_file:
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        args.results_file.write_text(json.dumps(results, indent=2))
        _LOGGER.info("Results saved to %s", args.results_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
