"""Tests for the LeRobot v3.0 -> v2.0 converter modality + info assembly."""

from __future__ import annotations

import json

import pytest
from convert_lerobot_v3_to_v2 import (
    UR10E_SINGLE_ARM_MODALITY,
    ModalityError,
    build_info_json,
    build_modality_json,
    load_modality_spec,
)


def _features(state_dim, action_dim):
    return {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action": {"dtype": "float32", "shape": [action_dim]},
    }


class TestModalityDefault:
    """The default modality spec targets the UR10e single-arm 7-DoF embodiment."""

    def test_default_ur10e_spec(self):
        modality = build_modality_json(_features(7, 7), load_modality_spec(None))
        assert modality["state"]["single_arm"] == {"start": 0, "end": 6}
        assert modality["state"]["gripper"] == {"start": 6, "end": 7}
        assert modality["action"]["single_arm"] == {"start": 0, "end": 6}
        assert modality["video"]["color"]["original_key"] == "observation.images.color"
        assert modality["video"]["color2"]["original_key"] == "observation.images.color2"

    def test_load_default_returns_independent_copy(self):
        spec = load_modality_spec(None)
        spec["state"]["single_arm"]["end"] = 99
        assert UR10E_SINGLE_ARM_MODALITY["state"]["single_arm"]["end"] == 6


class TestModalityValidation:
    """The spec must cover exactly the dataset state/action dimensions."""

    def test_state_dim_mismatch_raises(self):
        with pytest.raises(ModalityError):
            build_modality_json(_features(14, 14), load_modality_spec(None))

    def test_action_dim_mismatch_raises(self):
        with pytest.raises(ModalityError):
            build_modality_json(_features(7, 14), load_modality_spec(None))


class TestModalityConfig:
    """A custom embodiment is supplied via --modality-config JSON."""

    def test_custom_dual_arm_config_from_file(self, tmp_path):
        dual_arm = {
            "state": {"left_arm": {"start": 0, "end": 7}, "right_arm": {"start": 7, "end": 14}},
            "action": {"left_arm": {"start": 0, "end": 7}, "right_arm": {"start": 7, "end": 14}},
            "video": {"color_0": {"original_key": "observation.images.color_0"}},
            "annotation": {"human.task_description": {"original_key": "task_index"}},
        }
        config_path = tmp_path / "dual_arm_modality.json"
        config_path.write_text(json.dumps(dual_arm))
        modality = build_modality_json(_features(14, 14), load_modality_spec(config_path))
        assert modality["state"]["right_arm"] == {"start": 7, "end": 14}
        assert modality["video"]["color_0"]["original_key"] == "observation.images.color_0"


class TestBuildInfoJson:
    """info.json is rewritten with the v2.0 codebase version and path templates."""

    def test_v2_info_fields(self):
        src_info = {
            "codebase_version": "v3.0",
            "robot_type": "ur10e",
            "total_episodes": 5,
            "total_frames": 1000,
            "total_tasks": 1,
            "total_videos": 10,
            "chunks_size": 1000,
            "fps": 15,
            "features": {"observation.state": {"shape": [7]}},
        }
        info = build_info_json(src_info, num_episodes=5, num_chunks=1)
        assert info["codebase_version"] == "v2.0"
        assert info["robot_type"] == "ur10e"
        assert info["total_chunks"] == 1
        assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        assert info["features"] == src_info["features"]
        assert info["splits"] == {"train": "0:5"}
