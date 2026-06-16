# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "isaac-gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T.git@n1.5-release",
#     "torch",
#     "transformers",
#     "decord",
#     "accelerate",
# ]
# ///
"""Finetune GR00T-N1.5-3B on the dual-arm UR5e LeRobot v2.1 dataset.

Container/OSMO-friendly dual-arm trainer for real recorded LeRobot v2.1 data
(``teradyne-dual-leader-follower``). Differs from ``train_gr00t_dual_arm.py``
in three ways:

* Excludes the currently-constant gripper dimensions (min == max == 0.0) from
  the GR00T ``state_keys``/``action_keys`` to avoid a divide-by-zero in
  ``StateActionTransform(min_max)``; re-add them once real gripper data is
  recorded.
* Supports multi-dataset mixture training via ``--dataset-manifest`` (a JSON
  manifest listing several LeRobot roots), falling back to ``--dataset`` for a
  single root.
* Uses the ``adamw_torch`` optimizer and env-driven save cadence
  (``SAVE_STEPS``, ``SAVE_TOTAL_LIMIT``).

The two container/OSMO behavior choices are unchanged:

1. ``DATASET_PATH`` defaults to the ``$DATASET_PATH`` env var (else ``/data``).
2. ``OUTPUT_DIR`` defaults to the ``$OUTPUT_DIR`` env var (else
   ``/output/checkpoints/gr00t_n15_ur5e_dual_arm``).

The ``--smoke-test`` flag exits 0 cleanly on a CUDA-less host (no GR00T import
required) and runs a single post-training inference pass on a CUDA host.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
    from gr00t.data.transform.base import ModalityTransform
    from gr00t.experiment.data_config import BaseDataConfig
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from transformers import TrainingArguments

DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output/checkpoints/gr00t_n15_ur5e_dual_arm"))
BASE_MODEL = "nvidia/GR00T-N1.5-3B"

_LOGGER = logging.getLogger(__name__)


def _build_data_config_class() -> type[BaseDataConfig]:
    """Return the ``Ur5eDualArmDataConfig`` class, importing GR00T lazily."""
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
    from gr00t.data.transform.video import (
        VideoColorJitter,
        VideoCrop,
        VideoResize,
        VideoToNumpy,
        VideoToTensor,
    )
    from gr00t.experiment.data_config import BaseDataConfig
    from gr00t.model.transforms import GR00TTransform

    class Ur5eDualArmDataConfig(BaseDataConfig):
        # Modality keys MUST line up with meta/modality.json in the dataset.
        # state/action are a flat 14-dim concatenated vector laid out as
        # [robot1_arm(6), robot1_gripper(1), robot2_arm(6), robot2_gripper(1)].
        # The gripper slices are currently constant (min == max == 0.0), so they
        # are excluded from training to avoid a divide-by-zero in
        # StateActionTransform(min_max). Re-add state.robot{1,2}_gripper /
        # action.robot{1,2}_gripper once real gripper data is recorded.
        video_keys: ClassVar[list[str]] = [
            "video.color_0",
            "video.color_1",
            "video.color_2",
            "video.color_3",
        ]
        state_keys: ClassVar[list[str]] = [
            "state.robot1_arm",
            "state.robot2_arm",
        ]
        action_keys: ClassVar[list[str]] = [
            "action.robot1_arm",
            "action.robot2_arm",
        ]
        # modality.json declares the annotation subkey as
        # human.action.task_description (original_key task_index).
        language_keys: ClassVar[list[str]] = ["annotation.human.action.task_description"]

        # 1-step observation; 16-step action lookahead (~1.07 s at 15 fps).
        observation_indices: ClassVar[list[int]] = [0]
        action_indices: ClassVar[list[int]] = list(range(16))

        def transform(self) -> ModalityTransform:
            transforms = [
                VideoToTensor(apply_to=self.video_keys),
                VideoCrop(apply_to=self.video_keys, scale=0.95),
                VideoResize(
                    apply_to=self.video_keys,
                    height=224,
                    width=224,
                    interpolation="linear",
                ),
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finetune GR00T-N1.5-3B on the dual-arm UR5e LeRobot dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help="Path to a single LeRobot v2.x dataset root.")
    parser.add_argument("--dataset-manifest", type=Path, default=None,
                        help="Path to a JSON manifest (from prepare_datasets.py) "
                             "listing multiple dataset roots to train on as a "
                             "mixture. Overrides --dataset when provided.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Directory for HuggingFace Trainer checkpoints.")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL,
                        help="HuggingFace model id or local path to the base model.")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=int(os.environ.get("SAVE_STEPS", "500")),
        help="Save a checkpoint every N steps (env: SAVE_STEPS).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
        help="Set to 0 on Windows to avoid pickling errors. On Linux/OSMO "
             "containers, 4 is a reasonable default; override via "
             "DATALOADER_NUM_WORKERS env var or --num-workers.",
    )
    parser.add_argument(
        "--video-backend",
        choices=("decord", "torchcodec", "torchvision_av"),
        default="decord",
        help="Video decoder backend; decord is preferred on Windows.",
    )
    parser.add_argument("--freeze-llm", action="store_true", default=True,
                        help="Freeze the language backbone (default: True).")
    parser.add_argument("--freeze-visual", action="store_true", default=True,
                        help="Freeze the visual backbone (default: True).")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Optional path to a checkpoint to resume from.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Skip training when CUDA is unavailable; otherwise run a single "
             "post-training inference forward pass on dataset[0].",
    )
    return parser


def _build_training_args(args: argparse.Namespace) -> TrainingArguments:
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(args.output),
        bf16=True,
        tf32=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=4 if args.num_workers > 0 else None,
        dataloader_persistent_workers=args.num_workers > 0,
        learning_rate=args.lr,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "5")),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
    )


def _patch_gr00t_get_language() -> None:
    """Patch ``LeRobotSingleDataset.get_language`` to handle string task values.

    Some LeRobot datasets store the annotation column as raw task-description
    strings rather than integer task indices that index into
    ``meta/tasks.jsonl``. The upstream implementation unconditionally calls
    ``.item()`` on the value and crashes with
    ``AttributeError: 'str' object has no attribute 'item'``.
    """
    import numpy as np
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
    _LOGGER.info("Patched gr00t.data.dataset.LeRobotSingleDataset.get_language for string task values")


def _load_dataset_specs(args: argparse.Namespace) -> list[dict]:
    """Resolve the list of (path, video_backend) dataset specs to train on.

    Prefers ``--dataset-manifest`` (multi-dataset mixture); falls back to the
    single ``--dataset`` path using the global ``--video-backend``.
    """
    import json

    if args.dataset_manifest is not None:
        with open(args.dataset_manifest) as manifest_file:
            manifest = json.load(manifest_file)
        specs = manifest.get("datasets", [])
        if not specs:
            raise ValueError(f"Manifest {args.dataset_manifest} lists no datasets")
        return [
            {"path": spec["path"], "video_backend": spec.get("video_backend", args.video_backend)}
            for spec in specs
        ]
    return [{"path": str(args.dataset), "video_backend": args.video_backend}]


def _build_single(spec: dict, cfg: BaseDataConfig) -> LeRobotSingleDataset:
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag

    return LeRobotSingleDataset(
        dataset_path=spec["path"],
        modality_configs=cfg.modality_config(),
        transforms=cfg.transform(),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend=spec["video_backend"],
    )


def _build_dataset(args: argparse.Namespace, cfg: BaseDataConfig) -> LeRobotSingleDataset | LeRobotMixtureDataset:
    from gr00t.data.dataset import LeRobotMixtureDataset

    _patch_gr00t_get_language()

    specs = _load_dataset_specs(args)
    datasets = []
    for spec in specs:
        _LOGGER.info("Loading dataset %s (video_backend=%s)", spec["path"], spec["video_backend"])
        datasets.append(_build_single(spec, cfg))

    if len(datasets) == 1:
        return datasets[0]

    # Equal weight (1.0) for every dataset -> all are "primary"; the mixture
    # then balances sampling by trajectory length across datasets.
    mixture = LeRobotMixtureDataset(
        data_mixture=[(ds, 1.0) for ds in datasets],
        mode="train",
        balance_dataset_weights=True,
        balance_trajectory_weights=True,
        seed=42,
    )
    _LOGGER.info("Built mixture of %d datasets:\n%s", len(datasets), str(mixture))
    return mixture


def _build_model(args: argparse.Namespace) -> GR00T_N1_5:
    from gr00t.model.gr00t_n1 import GR00T_N1_5

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        tune_llm=not args.freeze_llm,
        tune_visual=not args.freeze_visual,
        tune_projector=True,
        tune_diffusion_model=True,
    )
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    return model


def _run_smoke_test(args: argparse.Namespace, cfg: BaseDataConfig, dataset: Any) -> int:
    from gr00t.data.schema import EmbodimentTag
    from gr00t.policy.policy import Gr00tPolicy

    checkpoints = sorted(
        args.output.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        _LOGGER.error("No checkpoint-* directory found under %s; cannot run smoke test.", args.output)
        return 1

    latest = checkpoints[-1]
    _LOGGER.info("Loading policy from %s", latest)
    policy = Gr00tPolicy(
        model_path=str(latest),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        device="cuda:0",
    )
    sample = dataset[0]
    action = policy.get_action(sample)
    _LOGGER.info(
        "Predicted action shapes: %s",
        {k: getattr(v, "shape", None) for k, v in action.items()},
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = create_parser().parse_args(argv)

    try:
        try:
            import torch
        except ImportError:
            _LOGGER.exception("Failed to import torch; install dependencies before running.")
            return 1

        if not torch.cuda.is_available():
            if args.smoke_test:
                _LOGGER.warning(
                    "CUDA is not available; smoke-test exiting cleanly without "
                    "loading GR00T or running training.",
                )
                return 0
            _LOGGER.error("CUDA is not available; finetuning GR00T-N1.5-3B requires a GPU.")
            return 1

        if args.dataset_manifest is not None:
            _LOGGER.info("Dataset manifest: %s", args.dataset_manifest)
        else:
            _LOGGER.info("Dataset path: %s", args.dataset)
        _LOGGER.info("Output dir:   %s", args.output)
        args.output.mkdir(parents=True, exist_ok=True)

        from gr00t.experiment.runner import TrainRunner

        data_config_cls = _build_data_config_class()
        cfg = data_config_cls()

        dataset = _build_dataset(args, cfg)
        _LOGGER.info("Dataset loaded: %d samples", len(dataset))

        model = _build_model(args)
        training_args = _build_training_args(args)

        runner = TrainRunner(
            train_dataset=dataset,
            model=model,
            training_args=training_args,
            resume_from_checkpoint=str(args.resume) if args.resume else False,
        )
        runner.train()

        if args.smoke_test:
            return _run_smoke_test(args, cfg, dataset)

        return 0
    except KeyboardInterrupt:
        _LOGGER.warning("Interrupted by user.")
        return 130
    except Exception:
        _LOGGER.exception("Training failed with an unhandled exception.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
