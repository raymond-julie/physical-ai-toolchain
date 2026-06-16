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
"""Finetune GR00T-N1.5-3B on the dual-arm UR5e LeRobot v2.0 dataset.

Container/OSMO-friendly trainer for the dual-arm UR5e embodiment (14-DoF,
4 cameras). Two behavior choices make it run unchanged inside an OSMO workflow
pod:

1. ``DATASET_PATH`` defaults to the ``$DATASET_PATH`` env var (else ``/data``).
   OSMO mounts the dataset payload at the path declared in the workflow YAML,
   and that mount path is the LeRobot v2.0 dataset root (``data/``, ``videos/``,
   ``meta/``).
2. ``OUTPUT_DIR`` defaults to the ``$OUTPUT_DIR`` env var (else
   ``/output/checkpoints/gr00t_n15_ur5e_dual_arm``).

Run via uv (resolves the PEP 723 metadata above):

    uv run train_gr00t_dual_arm.py

Or directly with a Python interpreter that already has the dependencies:

    python train_gr00t_dual_arm.py

GPU memory: roughly 24 GB at ``--batch-size 16`` with ``gradient_checkpointing``;
drop to 8 (or lower) on CUDA OOM.

The ``--smoke-test`` flag has dual purpose: on a CUDA-less host it logs a warning
and exits 0 cleanly (no GR00T import required, so ``--help`` and CPU validation
work without the package installed); on a CUDA host with an existing checkpoint
under ``--output`` it loads ``Gr00tPolicy`` and runs one inference pass on
``dataset[0]``, printing the predicted action shapes.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.transform.base import ModalityTransform
    from gr00t.experiment.data_config import BaseDataConfig
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from transformers import TrainingArguments

DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output/checkpoints/gr00t_n15_ur5e_dual_arm"))
BASE_MODEL = "nvidia/GR00T-N1.5-3B"

_LOGGER = logging.getLogger(__name__)


def _build_data_config_class() -> type[BaseDataConfig]:
    """Return the ``Ur5eDualArmDataConfig`` class, importing GR00T lazily.

    Defining the class inside this helper keeps every ``isaac-gr00t`` import out
    of module load so ``--help`` and the CPU-only ``--smoke-test`` exit paths do
    not require the package to be installed.
    """
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
                        help="Path to the LeRobot v2.0 dataset root.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Directory for HuggingFace Trainer checkpoints.")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL,
                        help="HuggingFace model id or local path to the base model.")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--save-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
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
        save_total_limit=10,
        # Emit TensorBoard event files under <output_dir>/runs/ every
        # logging_steps so the training tracker can read live metrics before the
        # first checkpoint flushes trainer_state.json.
        report_to="tensorboard",
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


def _build_dataset(args: argparse.Namespace, cfg: BaseDataConfig) -> LeRobotSingleDataset:
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag

    _patch_gr00t_get_language()

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
                _LOGGER.warning(
                    "CUDA is not available; smoke-test exiting cleanly without "
                    "loading GR00T or running training.",
                )
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
