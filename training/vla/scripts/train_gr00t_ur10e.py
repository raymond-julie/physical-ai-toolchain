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
"""Finetune GR00T-N1.5-3B on the single-arm UR10e LeRobot v2.0 dataset.

Single-arm sibling of ``train_gr00t_dual_arm.py``; consumes the
``hybrid-hack-vla-train-full`` dataset after the v3->v2 conversion. The two
container/OSMO-friendly behavior choices are unchanged:

1. ``DATASET_PATH`` defaults to the ``$DATASET_PATH`` env var (else ``/data``).
2. ``OUTPUT_DIR`` defaults to the ``$OUTPUT_DIR`` env var (else
   ``/output/checkpoints/gr00t_n15_ur10e_single_arm``).

Embodiment differences from the dual-arm trainer:

* 7-DoF state/action: 6 arm joints + 1 gripper position. The companion
  ``meta/modality.json`` splits these into ``single_arm`` (idx 0:6) and
  ``gripper`` (idx 6:7), so the GR00T ``state_keys``/``action_keys`` are
  ``state.single_arm``+``state.gripper`` (and the action equivalents).
* Two RGB cameras (``observation.images.color`` and
  ``observation.images.color2``) surfaced as ``video.color`` and
  ``video.color2``.
* Language annotation is the LeRobot ``task_index`` column rebound to
  ``annotation.human.task_description`` via modality.json.

Run via uv (resolves the PEP 723 metadata above):

    uv run train_gr00t_ur10e.py

Or directly with a Python interpreter that already has the dependencies:

    python train_gr00t_ur10e.py
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.transform.base import ModalityTransform
    from gr00t.experiment.data_config import BaseDataConfig
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from transformers import TrainingArguments

DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output/checkpoints/gr00t_n15_ur10e_single_arm"))
BASE_MODEL = "nvidia/GR00T-N1.5-3B"

_LOGGER = logging.getLogger(__name__)


def _build_data_config_class() -> type[BaseDataConfig]:
    """Return the ``Ur10eSingleArmDataConfig`` class, importing GR00T lazily."""
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

    class Ur10eSingleArmDataConfig(BaseDataConfig):
        # Match the modality.json video keys ("color", "color2"); GR00T prefixes
        # them with "video." when surfacing to data transforms.
        video_keys: ClassVar[list[str]] = [
            "video.color",
            "video.color2",
        ]
        state_keys: ClassVar[list[str]] = [
            "state.single_arm",
            "state.gripper",
        ]
        action_keys: ClassVar[list[str]] = [
            "action.single_arm",
            "action.gripper",
        ]
        # The annotation key in modality.json is "human.task_description"; GR00T
        # surfaces it as "annotation.human.task_description".
        language_keys: ClassVar[list[str]] = ["annotation.human.task_description"]

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

    return Ur10eSingleArmDataConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finetune GR00T-N1.5-3B on the single-arm UR10e LeRobot dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help="Path to the LeRobot v2.0 dataset root.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Directory for HuggingFace Trainer checkpoints.")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL,
                        help="HuggingFace model id or local path to the base model.")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
    )
    parser.add_argument(
        "--video-backend",
        choices=("decord", "torchcodec", "torchvision_av"),
        default="decord",
    )
    parser.add_argument("--freeze-llm", action="store_true", default=True)
    parser.add_argument("--freeze-visual", action="store_true", default=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
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
        save_steps=500,
        save_total_limit=5,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
    )


def _build_dataset(args: argparse.Namespace, cfg: BaseDataConfig) -> LeRobotSingleDataset:
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag

    return LeRobotSingleDataset(
        dataset_path=str(args.dataset),
        modality_configs=cfg.modality_config(),
        transforms=cfg.transform(),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend=args.video_backend,
    )


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


def _run_smoke_test(args: argparse.Namespace, cfg: BaseDataConfig, dataset: LeRobotSingleDataset) -> int:
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
                _LOGGER.warning("CUDA is not available; smoke-test exiting cleanly.")
                return 0
            _LOGGER.error("CUDA is not available; finetuning GR00T-N1.5-3B requires a GPU.")
            return 1

        _LOGGER.info("Dataset path: %s", args.dataset)
        _LOGGER.info("Output dir:   %s", args.output)
        args.output.mkdir(parents=True, exist_ok=True)

        from gr00t.experiment.runner import TrainRunner

        data_config_cls = _build_data_config_class()
        cfg = data_config_cls()

        dataset = _build_dataset(args, cfg)
        _LOGGER.info("Dataset loaded: %d samples from %s", len(dataset), args.dataset)

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
