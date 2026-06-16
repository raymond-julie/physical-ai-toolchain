"""Tests for the LeRobot recorder pure helpers (no ROS, no hardware).

Covers episode framing (``classify_episode``), joint reordering
(``reorder_joints``), and uint16 depth packing (``pack_depth_u16``). numpy is
real; cv2 is exercised only via the import-absent crop fallback.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from lerobot_recorder_node import (
    JOINT_NAMES,
    EpisodeDecision,
    classify_episode,
    pack_depth_u16,
    reorder_joints,
)


class TestClassifyEpisode:
    def test_zero_frames_is_discard_empty(self) -> None:
        assert classify_episode(0, min_episode_frames=5) is EpisodeDecision.DISCARD_EMPTY

    def test_below_minimum_is_discard_short(self) -> None:
        assert classify_episode(3, min_episode_frames=5) is EpisodeDecision.DISCARD_SHORT

    def test_at_or_above_minimum_is_save(self) -> None:
        assert classify_episode(5, min_episode_frames=5) is EpisodeDecision.SAVE
        assert classify_episode(50, min_episode_frames=5) is EpisodeDecision.SAVE

    def test_decision_values_are_stable_strings(self) -> None:
        assert EpisodeDecision.DISCARD_EMPTY == "discard_empty"
        assert EpisodeDecision.DISCARD_SHORT == "discard_short"
        assert EpisodeDecision.SAVE == "save"


class TestReorderJoints:
    def test_empty_inputs_return_none(self) -> None:
        assert reorder_joints([], [], JOINT_NAMES) is None
        assert reorder_joints(["a"], [], JOINT_NAMES) is None

    def test_named_joints_are_reordered_to_canonical_order(self) -> None:
        names = ["elbow_joint", "shoulder_pan_joint"]
        positions = [1.0, 2.0]
        result = reorder_joints(names, positions, ["shoulder_pan_joint", "elbow_joint"])
        assert result is not None
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [2.0, 1.0])

    def test_unknown_names_with_enough_positions_pass_through(self) -> None:
        result = reorder_joints(["a", "b", "c"], [5.0, 6.0, 7.0], ["x", "y"])
        assert result is not None
        np.testing.assert_allclose(result, [5.0, 6.0])

    def test_unknown_names_with_too_few_positions_return_none(self) -> None:
        assert reorder_joints(["a"], [5.0], ["x", "y"]) is None


class TestPackDepthU16:
    def test_high_and_low_bytes_are_split_per_pixel(self) -> None:
        depth = np.array([[0x0102, 0x00FF]], dtype=np.uint16)
        out = pack_depth_u16(depth, height=1, width=2)
        assert out.shape == (1, 2, 3)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out[0, 0], [0x01, 0x02, 0])
        np.testing.assert_array_equal(out[0, 1], [0x00, 0xFF, 0])

    def test_3d_depth_uses_first_channel(self) -> None:
        depth = np.array([[[0x0203], [0x0001]]], dtype=np.uint16)
        out = pack_depth_u16(depth, height=1, width=2)
        np.testing.assert_array_equal(out[0, 0], [0x02, 0x03, 0])
        np.testing.assert_array_equal(out[0, 1], [0x00, 0x01, 0])

    def test_float_depth_is_cast_to_uint16(self) -> None:
        depth = np.array([[258.0]], dtype=np.float32)  # 0x0102
        out = pack_depth_u16(depth, height=1, width=1)
        np.testing.assert_array_equal(out[0, 0], [0x01, 0x02, 0])

    def test_shape_mismatch_crops_when_cv2_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force the ImportError branch so packing falls back to a crop.
        monkeypatch.setitem(sys.modules, "cv2", None)
        depth = np.array(
            [[0x0101, 0x0102, 0x0103], [0x0201, 0x0202, 0x0203]],
            dtype=np.uint16,
        )
        out = pack_depth_u16(depth, height=2, width=2)
        assert out.shape == (2, 2, 3)
        np.testing.assert_array_equal(out[0, 0], [0x01, 0x01, 0])
        np.testing.assert_array_equal(out[1, 1], [0x02, 0x02, 0])
