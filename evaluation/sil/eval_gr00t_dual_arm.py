# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "isaac-gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T.git@n1.5-release",
#     "torch",
#     "transformers",
#     "decord",
#     "numpy",
# ]
# ///
"""Offline evaluation of a GR00T-N1.5-3B checkpoint on held-out episodes.

Loads the trained dual-arm UR5e policy from a checkpoint directory, runs
inference on a subset of episodes, and reports per-joint MSE between predicted
and ground-truth actions. This is the GR00T (Isaac, PyTorch/HuggingFace
``Trainer``) counterpart to the LeRobot ACT/Diffusion evaluator in
``scripts/batch-lerobot-eval.py``; both compute offline replay error, but this
script consumes the GR00T-flavored LeRobot v2.0 dataset and the
``Gr00tPolicy`` action interface.

The data config mirrors ``training/vla/scripts/train_gr00t_dual_arm.py`` but
drops the training-time crop/jitter augmentations so evaluation runs on
unperturbed frames.

Usage:
    uv run eval_gr00t_dual_arm.py --checkpoint /outputs/.../checkpoint-100000 \\
        --dataset /data --holdout-episodes 50,51,52,53,54,55,56

    # Or hold out the last N episodes automatically:
    uv run eval_gr00t_dual_arm.py --checkpoint /outputs/.../checkpoint-100000 \\
        --dataset /data --holdout-last 7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from gr00t.data.transform.base import ModalityTransform
    from gr00t.experiment.data_config import BaseDataConfig

DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs/gr00t-n15-dual-arm-combined-sessions"))

_LOGGER = logging.getLogger(__name__)


def _build_data_config_class() -> type[BaseDataConfig]:
    """Return the ``Ur5eDualArmDataConfig`` class, importing GR00T lazily.

    Defining the class inside this helper keeps every ``isaac-gr00t`` import out
    of module load, so ``--help`` works without the package installed.
    """
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
    from gr00t.data.transform.video import VideoResize, VideoToNumpy, VideoToTensor
    from gr00t.experiment.data_config import BaseDataConfig
    from gr00t.model.transforms import GR00TTransform

    class Ur5eDualArmDataConfig(BaseDataConfig):
        video_keys: ClassVar[list[str]] = [
            "video.color_0",
            "video.color_1",
            "video.color_2",
            "video.color_3",
        ]
        state_keys: ClassVar[list[str]] = [
            "state.robot1_arm",
            "state.robot1_gripper",
            "state.robot2_arm",
            "state.robot2_gripper",
        ]
        action_keys: ClassVar[list[str]] = [
            "action.robot1_arm",
            "action.robot1_gripper",
            "action.robot2_arm",
            "action.robot2_gripper",
        ]
        language_keys: ClassVar[list[str]] = ["annotation.human.action.task_description"]

        observation_indices: ClassVar[list[int]] = [0]
        action_indices: ClassVar[list[int]] = list(range(16))

        def transform(self) -> ModalityTransform:
            transforms = [
                VideoToTensor(apply_to=self.video_keys),
                VideoResize(
                    apply_to=self.video_keys,
                    height=224,
                    width=224,
                    interpolation="linear",
                ),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: "min_max" for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={k: "min_max" for k in self.action_keys},
                ),
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                GR00TTransform(
                    state_horizon=len(self.observation_indices),
                    action_horizon=len(self.action_indices),
                    max_state_dim=64,
                    max_action_dim=32,
                ),
            ]
            return ComposedModalityTransform(transforms=transforms)

    return Ur5eDualArmDataConfig


def _patch_gr00t_get_language() -> None:
    """Patch ``LeRobotSingleDataset.get_language`` to handle string task values.

    Mirrors the patch in ``train_gr00t_dual_arm.py``: datasets that store the
    annotation column as raw task strings (rather than integer task indices)
    otherwise crash on the upstream ``.item()`` call.
    """
    from gr00t.data.dataset import LeRobotSingleDataset

    def patched_get_language(self: Any, trajectory_id: Any, key: str, base_index: int) -> list[str]:
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)

        assert key.startswith("annotation."), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None
        assert subkey in annotation_meta
        original_key = annotation_meta[subkey].original_key
        if original_key is None:
            original_key = key

        results = []
        task_indices = []
        use_strings = False
        for i in range(len(step_indices)):
            value = self.curr_traj_data[original_key][step_indices[i]]
            if isinstance(value, str):
                use_strings = True
                results.append(value)
            else:
                try:
                    task_indices.append(int(value.item()))
                except AttributeError:
                    task_indices.append(int(value))
        if use_strings:
            return results
        return self.tasks.loc[task_indices]["task"].tolist()

    LeRobotSingleDataset.get_language = patched_get_language


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    """Return the checkpoint path, auto-detecting the latest when unspecified."""
    if args.checkpoint:
        return args.checkpoint

    checkpoints = sorted(
        args.output.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        _LOGGER.error("No checkpoints found in %s", args.output)
        sys.exit(1)
    latest = checkpoints[-1]
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


def _flatten_action(action: Any) -> np.ndarray:
    """Flatten a per-step action (dict of arrays or a single array) to 1-D."""
    if isinstance(action, dict):
        return np.concatenate([np.asarray(v).flatten() for v in action.values()])
    return np.asarray(action).flatten()


def _evaluate(
    policy: Any,
    dataset: Any,
    holdout_indices: list[int],
    stride: int,
) -> dict[str, Any]:
    """Run evaluation on held-out frames and compute per-joint MSE."""
    all_errors: list[np.ndarray] = []
    per_episode: dict[int, dict[str, Any]] = {}

    for ep_idx in holdout_indices:
        ep_start = None
        ep_end = None
        for i in range(len(dataset)):
            sample = dataset[i]
            ei = sample.get("episode_index") if hasattr(sample, "get") else getattr(sample, "episode_index", None)
            if ei == ep_idx:
                if ep_start is None:
                    ep_start = i
                ep_end = i
            elif ep_start is not None:
                break

        if ep_start is None:
            _LOGGER.warning("Episode %d not found in dataset, skipping", ep_idx)
            continue

        ep_errors: list[np.ndarray] = []
        for i in range(ep_start, ep_end + 1, stride):
            sample = dataset[i]
            gt_action = sample.get("action", None)
            if gt_action is None:
                continue

            gt_flat = _flatten_action(gt_action)
            pred_flat = _flatten_action(policy.get_action(sample))

            min_len = min(len(gt_flat), len(pred_flat))
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

    return {
        "overall_mse": float(np.mean(all_errors)),
        "per_dim_mse": np.mean(all_errors, axis=0).tolist(),
        "total_frames": sum(e["frames"] for e in per_episode.values()),
        "episodes_evaluated": len(per_episode),
        "per_episode": per_episode,
    }


def _print_summary(checkpoint_path: Path, results: dict[str, Any]) -> None:
    overall = results.get("overall_mse")
    overall_str = f"{overall:.6f}" if overall is not None else "N/A"

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (GR00T-N1.5-3B)")
    print("=" * 60)
    print(f"Checkpoint:         {checkpoint_path.name}")
    print(f"Episodes evaluated: {results.get('episodes_evaluated', 0)}")
    print(f"Total frames:       {results.get('total_frames', 0)}")
    print(f"Overall MSE:        {overall_str}")
    print("=" * 60)

    if "per_dim_mse" in results:
        print("\nPer-dimension MSE:")
        for i, mse in enumerate(results["per_dim_mse"]):
            print(f"  dim[{i:2d}]: {mse:.6f}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a GR00T-N1.5-3B checkpoint on held-out episodes.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to checkpoint dir. Auto-detects latest if omitted.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Output dir containing checkpoints (for auto-detection).")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help="Path to the LeRobot v2.0 dataset root.")
    parser.add_argument("--holdout-episodes", type=str, default=None,
                        help="Comma-separated episode indices to evaluate on.")
    parser.add_argument("--holdout-last", type=int, default=7,
                        help="Hold out the last N episodes (default: 7).")
    parser.add_argument("--stride", type=int, default=10,
                        help="Evaluate every Nth frame per episode (default: 10).")
    parser.add_argument("--video-backend", choices=("decord", "torchcodec", "torchvision_av"),
                        default="decord")
    parser.add_argument("--results-file", type=Path, default=None,
                        help="Save results JSON to this path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = create_parser().parse_args(argv)

    try:
        import torch
    except ImportError:
        _LOGGER.exception("Failed to import torch; install dependencies before running.")
        return 1

    if not torch.cuda.is_available():
        _LOGGER.error("CUDA is not available; GR00T evaluation requires a GPU.")
        return 1

    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.policy.policy import Gr00tPolicy

    _patch_gr00t_get_language()

    checkpoint_path = _resolve_checkpoint(args)
    holdout = _get_holdout_episodes(args)
    _LOGGER.info("Checkpoint: %s", checkpoint_path)
    _LOGGER.info("Holdout episodes: %s", holdout)

    cfg = _build_data_config_class()()

    dataset = LeRobotSingleDataset(
        dataset_path=str(args.dataset),
        modality_configs=cfg.modality_config(),
        transforms=cfg.transform(),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend=args.video_backend,
    )
    _LOGGER.info("Dataset loaded: %d samples", len(dataset))

    policy = Gr00tPolicy(
        model_path=str(checkpoint_path),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        device="cuda:0",
    )

    results = _evaluate(policy, dataset, holdout, args.stride)
    _print_summary(checkpoint_path, results)

    if args.results_file:
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        args.results_file.write_text(json.dumps(results, indent=2))
        _LOGGER.info("Results saved to %s", args.results_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
