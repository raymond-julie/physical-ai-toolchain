"""GR00T modality config for the Schaeffler bimanual UR5e (12-D action/state).

Layout (matches `meta/modality.json` slice indices in the converted dataset):

    state / action:
      right_arm  [0:6]   6-DOF UR5e joint state, RELATIVE on the action side
      left_arm   [6:12]  6-DOF UR5e joint state, RELATIVE on the action side

    video:
      front        scene camera   (observation.images.d405_stationary_r_0)
      wrist_left   wrist camera   (observation.images.d405_stationary_l_1)
      wrist_right  wrist camera   (observation.images.d405_stationary_l_2)

The 16-step action horizon (`delta_indices=range(0, 16)`) matches the SO100
reference example and corresponds to ~0.53 s at 30 Hz. Both arms are declared
RELATIVE so they line up with `launch_finetune.py`'s hard-coded
`use_relative_action=True`. The downstream controller is responsible for
integrating the predicted arm deltas onto the current joint state.

The Schaeffler dataset has no gripper channels; joint order is right-arm first
(R_joint_1..6) followed by left-arm (L_joint_1..6).

The file is loaded by GR00T's launcher via `--modality-config-path`. The
AzureML entry script copies it into the Isaac-GR00T clone at runtime
(under `examples/UR5eBimanual/`) so the relative import inside the
launcher resolves.
"""

from __future__ import annotations

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

UR5E_BIMANUAL_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist_left", "wrist_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right_arm", "left_arm"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["right_arm", "left_arm"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(UR5E_BIMANUAL_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
