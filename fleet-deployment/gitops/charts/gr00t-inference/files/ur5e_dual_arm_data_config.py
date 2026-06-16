"""GR00T N1.5 data config for the dual-arm UR5e fine-tune (checkpoint-40000).

Extracted verbatim from the training script (train_gr00t_dual_arm.py,
``Ur5eDualArmDataConfig``) so inference uses the EXACT same modality layout and
transforms the checkpoint was trained with. Any drift here (keys, normalization
modes, horizons) yields wrong actions on the real robot.

GR00T N1.5's ``scripts/inference_service.py`` resolves this via
``--data-config ur5e_dual_arm_data_config:Ur5eDualArmDataConfig``;
``load_data_config`` adds the process working directory (the server's WORKDIR
``/workspace``) to ``sys.path`` and imports this module, so it is mounted there.
"""

from __future__ import annotations

from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
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
    # State/action are a flat 14-dim concatenated vector laid out as
    # [robot1_arm(6), robot1_gripper(1), robot2_arm(6), robot2_gripper(1)].
    # The gripper slices are constant (min == max == 0.0) in this dataset, so
    # they are excluded from training/inference to avoid a divide-by-zero in
    # StateActionTransform(min_max). Re-add the gripper keys once real gripper
    # data is recorded (and retrain).
    video_keys = [
        "video.color_0",
        "video.color_1",
        "video.color_2",
        "video.color_3",
    ]
    state_keys = [
        "state.robot1_arm",
        "state.robot2_arm",
    ]
    action_keys = [
        "action.robot1_arm",
        "action.robot2_arm",
    ]
    language_keys = ["annotation.human.action.task_description"]

    # 1-step observation; 16-step action lookahead.
    observation_indices = [0]
    action_indices = list(range(16))

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
