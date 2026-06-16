"""UR10e single-arm policy + LeRobot data config for openpi fine-tuning.

Mirrors the structure of ``openpi.policies.libero_policy`` and
``openpi.training.config.LeRobotLiberoDataConfig``. Adapts the merged LeRobot
v2.x dataset (see ``data-management/tools/merge_lerobot_sessions.py``):

  - robot_type: ur10e
  - fps:        15
  - state[7]:   6 arm joint positions + 1 gripper (0..1)
  - action[7]:  6 arm joint targets + 1 gripper target
  - images:     observation.images.color      (third-person)
                observation.images.color2     (secondary view; used as wrist)
  - prompt:     read from per-episode ``task`` string in meta/tasks.jsonl

This module is imported by ``train_openpi_ur10e.py`` and the openpi train scripts
after it is copied into ``openpi/src/openpi/policies/``.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING, Any, override

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# libero_policy is imported for its config-registration side effect.
from openpi.policies import libero_policy  # noqa: F401
from openpi.training import config as openpi_config
from openpi.training import weight_loaders

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig

# Action chunk size used by both data loading and the model config.
DEFAULT_ACTION_HORIZON = 16
# Raw state / action width in the LeRobot dataset.
RAW_ACTION_DIM = 7


def _parse_image(image: Any) -> np.ndarray:
    """Convert a LeRobot frame tensor to HWC uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Ur10eInputs(transforms.DataTransformFn):
    """Map repacked dataset rows into the dict the model expects.

    Expected input keys (after the repack transform below):
        observation/image          - third-person RGB (HWC or CHW)
        observation/wrist_image    - secondary RGB (HWC or CHW); zeros are OK
        observation/state          - shape (7,)
        actions                    - shape (action_horizon, 7), training only
        prompt                     - language string, optional
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # No right wrist camera on this rig; pad with zeros.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # pi0-FAST gets True masks on padded images, pi0 / pi0.5 get False.
                "right_wrist_0_rgb": (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                ),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Ur10eOutputs(transforms.DataTransformFn):
    """Slice padded model actions back down to the 7-DoF UR10e action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :RAW_ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class LeRobotUr10eDataConfig(openpi_config.DataConfigFactory):
    """DataConfigFactory for the merged UR10e LeRobot dataset.

    Keep ``extra_delta_transform=True`` when the dataset stores ABSOLUTE joint
    targets in ``action`` (which it does); the gripper (last dim) stays absolute.
    """

    extra_delta_transform: bool = True
    default_prompt: str | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        # Step 1: repack — rename LeRobot keys into the schema Ur10eInputs reads.
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.color",
                        "observation/wrist_image": "observation.images.color2",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Step 2: dataset-specific input / output transforms.
        data_transforms = transforms.Group(
            inputs=[Ur10eInputs(model_type=model_config.model_type)],
            outputs=[Ur10eOutputs()],
        )

        # Step 3: optional absolute -> delta action conversion (joints only).
        if self.extra_delta_transform:
            # 6 joint dims get delta-encoded; gripper (-1 -> mask False) stays absolute.
            delta_action_mask = transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_action_mask)],
                outputs=[transforms.AbsoluteActions(delta_action_mask)],
            )

        # Step 4: standard model transforms (image resize, tokenize, pad).
        model_transforms = openpi_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


def build_train_configs(
    repo_id: str,
    *,
    exp_name: str,
    num_train_steps: int = 30_000,
    batch_size: int = 32,
    save_interval: int = 2_000,
    keep_period: int | None = 10_000,
    lora: bool = False,
    pi05: bool = True,
    prompt_from_task: bool = True,
    default_prompt: str | None = None,
) -> TrainConfig:
    """Return a TrainConfig for fine-tuning pi0 / pi0.5 on the UR10e dataset.

    ``repo_id`` is the path or HF repo of the LeRobot dataset (e.g.
    ``/data/combined-sessions`` or ``your_org/ur10e_sessions``).
    """
    from openpi.models import pi0_config

    if pi05:
        config_name = "pi05_ur10e_lora" if lora else "pi05_ur10e"
        model = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=DEFAULT_ACTION_HORIZON,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        )
        base_weights = "gs://openpi-assets/checkpoints/pi05_base/params"
    else:
        config_name = "pi0_ur10e_lora" if lora else "pi0_ur10e"
        model = pi0_config.Pi0Config(
            action_horizon=DEFAULT_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        )
        base_weights = "gs://openpi-assets/checkpoints/pi0_base/params"

    data = LeRobotUr10eDataConfig(
        repo_id=repo_id,
        base_config=openpi_config.DataConfig(prompt_from_task=prompt_from_task),
        default_prompt=default_prompt,
        extra_delta_transform=True,
    )

    train_kwargs = {
        "name": config_name,
        "exp_name": exp_name,
        "model": model,
        "data": data,
        "weight_loader": weight_loaders.CheckpointWeightLoader(base_weights),
        "num_train_steps": num_train_steps,
        "batch_size": batch_size,
        "save_interval": save_interval,
        "keep_period": keep_period,
    }

    if lora:
        train_kwargs["freeze_filter"] = model.get_freeze_filter()
        train_kwargs["ema_decay"] = None

    return openpi_config.TrainConfig(**train_kwargs)


def register(config: TrainConfig) -> None:
    """Append a TrainConfig to openpi's lookup dict so ``scripts/train.py`` finds it."""
    openpi_config._CONFIGS_DICT[config.name] = config  # type: ignore[attr-defined]
    openpi_config._CONFIGS.append(config)  # type: ignore[attr-defined]


__all__ = [
    "DEFAULT_ACTION_HORIZON",
    "RAW_ACTION_DIM",
    "LeRobotUr10eDataConfig",
    "Ur10eInputs",
    "Ur10eOutputs",
    "build_train_configs",
    "register",
]
