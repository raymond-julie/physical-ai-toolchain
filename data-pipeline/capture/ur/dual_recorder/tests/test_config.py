"""Tests for ur_dual_recorder config_v3 parsing (no hardware).

Covers device parsing (arms/cameras), leader/follower pairing, the defaults <
app.yaml overlay merge, side inference, and the error paths for a missing config
or a topology with no usable pairs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ur_dual_recorder.config import (
    DEFAULT_APP_SETTINGS,
    ArmConfig,
    CameraConfig,
    DualRecorderConfigError,
    _build_pairs,
    _deep_merge,
    _infer_side,
    _parse_devices,
    load_app_settings,
    load_config,
)

_CONFIG_V3 = """
recordings_dir: /data/recordings
copy_target_dir: /mnt/target
devices:
  arm_left_leader:
    type: arm
    name: Left Leader
    ip: 192.168.1.10
    mode: leader
    model: ur_ps5
  arm_left_follower:
    type: arm
    name: Left Follower
    ip: 192.168.1.11
    mode: follower
  arm_right_leader:
    type: arm
    ip: 192.168.1.12
    mode: leader
  arm_right_follower:
    type: arm
    ip: 192.168.1.13
    mode: follower
  cam_high:
    type: camera
    name: High Cam
    model: Gemini 335
    serial: CV3H4600001E
    specs:
      default:
        resolution: 848x480
        fps: 30
  arm_no_ip:
    type: arm
    mode: follower
"""


def _write_config(tmp_path: Path, text: str = _CONFIG_V3) -> Path:
    path = tmp_path / "config_v3.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestInferSide:
    def test_left_and_right_are_detected(self) -> None:
        assert _infer_side("arm_left_leader") == "left"
        assert _infer_side("arm_RIGHT_follower") == "right"

    def test_unsided_id_returns_empty(self) -> None:
        assert _infer_side("cam_high") == ""


class TestDeepMerge:
    def test_override_replaces_nested_leaf_only(self) -> None:
        merged = _deep_merge(DEFAULT_APP_SETTINGS, {"gripper": {"closed_threshold": 200}})
        assert merged["gripper"]["closed_threshold"] == 200
        # Sibling keys under gripper survive the merge.
        assert merged["gripper"]["port"] == DEFAULT_APP_SETTINGS["gripper"]["port"]

    def test_source_is_not_mutated(self) -> None:
        original = DEFAULT_APP_SETTINGS["gripper"]["closed_threshold"]
        _deep_merge(DEFAULT_APP_SETTINGS, {"gripper": {"closed_threshold": 7}})
        assert DEFAULT_APP_SETTINGS["gripper"]["closed_threshold"] == original


class TestParseDevices:
    def test_arms_and_cameras_are_split_by_type(self, tmp_path: Path) -> None:
        import yaml

        raw = yaml.safe_load(_CONFIG_V3)
        arms, cameras = _parse_devices(raw["devices"])
        assert set(cameras) == {"cam_high"}
        assert isinstance(arms["arm_left_leader"], ArmConfig)
        assert isinstance(cameras["cam_high"], CameraConfig)

    def test_arm_without_ip_is_skipped(self) -> None:
        import yaml

        arms, _cameras = _parse_devices(yaml.safe_load(_CONFIG_V3)["devices"])
        assert "arm_no_ip" not in arms

    def test_camera_specs_are_read(self) -> None:
        import yaml

        _arms, cameras = _parse_devices(yaml.safe_load(_CONFIG_V3)["devices"])
        cam = cameras["cam_high"]
        assert cam.serial == "CV3H4600001E"
        assert cam.resolution == "848x480"
        assert cam.fps == 30


class TestBuildPairs:
    def test_pairs_are_matched_by_side_and_sorted(self) -> None:
        import yaml

        arms, _cameras = _parse_devices(yaml.safe_load(_CONFIG_V3)["devices"])
        pairs = _build_pairs(arms)
        assert [p.side for p in pairs] == ["left", "right"]
        assert pairs[0].leader.device_id == "arm_left_leader"
        assert pairs[0].follower.device_id == "arm_left_follower"

    def test_leader_without_follower_raises(self) -> None:
        arms = {
            "arm_left_leader": ArmConfig(
                device_id="arm_left_leader", name="L", ip="x", mode="leader", side="left"
            )
        }
        with pytest.raises(DualRecorderConfigError, match="no matching follower"):
            _build_pairs(arms)

    def test_no_pairs_raises(self) -> None:
        with pytest.raises(DualRecorderConfigError, match="No leader/follower pairs"):
            _build_pairs({})


class TestLoadAppSettings:
    def test_none_returns_defaults_copy(self) -> None:
        settings = load_app_settings(None)
        assert settings["gripper"]["closed_threshold"] == 128
        assert settings is not DEFAULT_APP_SETTINGS

    def test_overlay_file_is_merged(self, tmp_path: Path) -> None:
        overlay = tmp_path / "app.yaml"
        overlay.write_text("gripper:\n  closed_threshold: 200\n", encoding="utf-8")
        settings = load_app_settings(str(overlay))
        assert settings["gripper"]["closed_threshold"] == 200

    def test_missing_overlay_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DualRecorderConfigError, match="app-config not found"):
            load_app_settings(str(tmp_path / "missing.yaml"))


class TestLoadConfig:
    def test_full_config_resolves(self, tmp_path: Path) -> None:
        config = load_config(str(_write_config(tmp_path)))
        assert set(config.cameras) == {"cam_high"}
        assert [p.side for p in config.pairs] == ["left", "right"]
        assert config.recordings_dir == "/data/recordings"
        assert config.copy_target_dir == "/mnt/target"
        assert config.gripper["closed_threshold"] == 128

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DualRecorderConfigError, match="config not found"):
            load_config(str(tmp_path / "nope.yaml"))
