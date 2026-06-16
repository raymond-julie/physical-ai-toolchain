"""Tests for camera_streamer configuration and camera selection."""

from __future__ import annotations

from pathlib import Path

import pytest
from camera_streamer.config import (
    DEFAULT_APP_SETTINGS,
    AppConfig,
    CameraStreamerConfigError,
    camera_list,
    load_app_settings,
    load_config,
)

_DEVICE_YAML = """
devices:
  cam_high:
    type: camera
    name: High Camera
    serial: "AY12345"
    model: Gemini 335
    specs:
      default:
        resolution: 1280x720
        fps: 15
  arm_left:
    type: arm
    name: Left Arm
  cam_low:
    type: camera
    name: Low Camera
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadConfigSelection:
    def test_selects_only_camera_devices(self, tmp_path: Path) -> None:
        cfg_path = _write(tmp_path / "config_v3.yaml", _DEVICE_YAML)
        config = load_config(str(cfg_path))
        assert set(config.cameras) == {"cam_high", "cam_low"}

    def test_camera_fields_from_specs(self, tmp_path: Path) -> None:
        cfg_path = _write(tmp_path / "config_v3.yaml", _DEVICE_YAML)
        config = load_config(str(cfg_path))
        high = config.cameras["cam_high"]
        assert high.name == "High Camera"
        assert high.serial == "AY12345"
        assert high.resolution == "1280x720"
        assert high.fps == 15

    def test_camera_defaults_when_specs_missing(self, tmp_path: Path) -> None:
        cfg_path = _write(tmp_path / "config_v3.yaml", _DEVICE_YAML)
        config = load_config(str(cfg_path))
        low = config.cameras["cam_low"]
        assert low.resolution == "848x480"
        assert low.fps == 30

    def test_camera_list_matches(self, tmp_path: Path) -> None:
        cfg_path = _write(tmp_path / "config_v3.yaml", _DEVICE_YAML)
        config = load_config(str(cfg_path))
        ids = sorted(cam.device_id for cam in camera_list(config))
        assert ids == ["cam_high", "cam_low"]

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CameraStreamerConfigError, match="not found"):
            load_config(str(tmp_path / "missing.yaml"))


class TestAppSettings:
    def test_defaults_when_no_overlay(self) -> None:
        settings = load_app_settings(None)
        assert settings["server"]["port"] == 8000
        assert settings["stream"]["jpeg_quality"] == 80

    def test_overlay_deep_merges(self, tmp_path: Path) -> None:
        overlay = _write(tmp_path / "app.yaml", "server:\n  port: 9100\nstream:\n  fps: 10\n")
        settings = load_app_settings(str(overlay))
        assert settings["server"]["port"] == 9100
        assert settings["server"]["host"] == "0.0.0.0"
        assert settings["stream"]["fps"] == 10
        assert settings["stream"]["jpeg_quality"] == 80

    def test_defaults_not_mutated(self, tmp_path: Path) -> None:
        overlay = _write(tmp_path / "app.yaml", "server:\n  port: 9100\n")
        load_app_settings(str(overlay))
        assert DEFAULT_APP_SETTINGS["server"]["port"] == 8000

    def test_missing_overlay_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CameraStreamerConfigError, match="not found"):
            load_app_settings(str(tmp_path / "missing.yaml"))


class TestDiscoveryFallback:
    def test_load_config_without_file_discovers_empty(self) -> None:
        config = load_config(None)
        assert isinstance(config, AppConfig)
        assert config.cameras == {}
        assert config.source_path is None
