"""UR5e dual-arm policy + LeRobot data config for openpi fine-tuning.

Adapts the merged dual-arm LeRobot v2.1 dataset (see
``data-management/tools/merge_lerobot_sessions.py``):

  - robot_type:   ur5e dual-arm
  - fps:          15
  - state[14]:    [r1_joint0..5, r1_gripper, r2_joint0..5, r2_gripper]
  - action[14]:   same layout as state
  - images:       observation.images.color_0  (third-person primary)
                  observation.images.color_1  (third-person secondary)
                  observation.images.color_2  (wrist / left)
                  observation.images.color_3  (wrist / right)
  - prompt:       per-episode task string (defaulted via --default-prompt)

pi0 models accept up to 3 camera views (base + left_wrist + right_wrist). The
4 cameras map as: color_0 -> base, color_2 -> left_wrist, color_3 -> right_wrist;
color_1 (the secondary third-person view) is dropped by default. Pass
``--use-secondary-base`` on the trainer to swap color_1 in for color_0 instead.

This module is imported by ``train_openpi_ur5e_dual_arm.py`` and the openpi train
scripts after it is copied into ``openpi/src/openpi/policies/``.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING, Any, override

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from openpi.training import config as openpi_config
from openpi.training import weight_loaders

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig

# Action chunk size used by both data loading and the model config.
DEFAULT_ACTION_HORIZON = 16
# Raw state / action width in the LeRobot dataset.
RAW_ACTION_DIM = 14


def _parse_image(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Ur5eDualArmInputs(transforms.DataTransformFn):
    """Map repacked dataset rows into the dict the model expects.

    Expected input keys (after the repack transform below):
        observation/image          - base RGB
        observation/wrist_image    - left wrist RGB (color_2)
        observation/wrist_image_2  - right wrist RGB (color_3)
        observation/state          - shape (14,)
        actions                    - shape (action_horizon, 14), training only
        prompt                     - language string, optional
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        wrist_image_2 = (
            _parse_image(data["observation/wrist_image_2"])
            if "observation/wrist_image_2" in data
            else np.zeros_like(base_image)
        )

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": wrist_image_2,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We have real data for every camera here, so always True.
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Ur5eDualArmOutputs(transforms.DataTransformFn):
    """Slice padded model actions back to the 14-DoF dual-arm space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :RAW_ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class LeRobotUr5eDualArmDataConfig(openpi_config.DataConfigFactory):
    """DataConfigFactory for the dual-arm LeRobot dataset.

    The dataset stores ABSOLUTE joint targets in ``action``. Keep
    ``extra_delta_transform=True`` (default) so the model trains on per-joint
    deltas; the two gripper dimensions stay absolute.
    """

    extra_delta_transform: bool = True
    default_prompt: str | None = None
    # Swap the "secondary" base camera (color_1) into the base slot instead of
    # color_0. Useful when color_0 is the less informative angle.
    use_secondary_base: bool = False

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        base_key = (
            "observation.images.color_1"
            if self.use_secondary_base
            else "observation.images.color_0"
        )

        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": base_key,
                        "observation/wrist_image": "observation.images.color_2",
                        "observation/wrist_image_2": "observation.images.color_3",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = transforms.Group(
            inputs=[Ur5eDualArmInputs(model_type=model_config.model_type)],
            outputs=[Ur5eDualArmOutputs()],
        )

        if self.extra_delta_transform:
            # 14-DoF action layout: [r1_j0..5, r1_grip, r2_j0..5, r2_grip].
            # Convert joint dims to deltas; keep both grippers absolute.
            #   make_bool_mask(6, -1, 6, -1) -> [T,T,T,T,T,T,F,T,T,T,T,T,T,F]
            delta_action_mask = transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_action_mask)],
                outputs=[transforms.AbsoluteActions(delta_action_mask)],
            )

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
    save_interval: int = 25_000,
    keep_period: int | None = 10_000,
    lora: bool = False,
    pi05: bool = True,
    prompt_from_task: bool = True,
    default_prompt: str | None = None,
    use_secondary_base: bool = False,
    assets_base_dir: str = "./assets",
    checkpoint_base_dir: str = "./checkpoints",
) -> TrainConfig:
    """Return a TrainConfig for fine-tuning pi0 / pi0.5 on the dual-arm UR5e dataset.

    ``repo_id`` is the path or HF repo of the LeRobot dataset (e.g.
    ``/data/combined-sessions``).
    """
    from openpi.models import pi0_config

    if pi05:
        config_name = "pi05_ur5e_dual_lora" if lora else "pi05_ur5e_dual"
        model = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=DEFAULT_ACTION_HORIZON,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        )
        base_weights = "gs://openpi-assets/checkpoints/pi05_base/params"
    else:
        config_name = "pi0_ur5e_dual_lora" if lora else "pi0_ur5e_dual"
        model = pi0_config.Pi0Config(
            action_horizon=DEFAULT_ACTION_HORIZON,
            paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        )
        base_weights = "gs://openpi-assets/checkpoints/pi0_base/params"

    data = LeRobotUr5eDualArmDataConfig(
        repo_id=repo_id,
        base_config=openpi_config.DataConfig(prompt_from_task=prompt_from_task),
        default_prompt=default_prompt,
        extra_delta_transform=True,
        use_secondary_base=use_secondary_base,
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
        "assets_base_dir": assets_base_dir,
        "checkpoint_base_dir": checkpoint_base_dir,
    }

    if lora:
        train_kwargs["freeze_filter"] = model.get_freeze_filter()
        train_kwargs["ema_decay"] = None

    return openpi_config.TrainConfig(**train_kwargs)


def register(config: TrainConfig) -> None:
    """Append a TrainConfig to openpi's lookup dict."""
    openpi_config._CONFIGS_DICT[config.name] = config  # type: ignore[attr-defined]
    openpi_config._CONFIGS.append(config)  # type: ignore[attr-defined]


__all__ = [
    "DEFAULT_ACTION_HORIZON",
    "RAW_ACTION_DIM",
    "LeRobotUr5eDualArmDataConfig",
    "Ur5eDualArmInputs",
    "Ur5eDualArmOutputs",
    "build_train_configs",
    "register",
]
