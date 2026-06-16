"""Tests for ur_dual_recorder camera selection (no Orbbec SDK, no network).

Covers resolution parsing, the serial -> MJPEG stream URL mapping (with
per-camera overrides), and the CameraManager source switch (orbbec devices vs.
shared MJPEG streams).
"""

from __future__ import annotations

from ur_dual_recorder.cameras import (
    CameraManager,
    MjpegCamera,
    OrbbecCamera,
    _parse_resolution,
    _stream_url,
)
from ur_dual_recorder.config import CameraConfig


def _camera(device_id: str = "cam_high", serial: str = "CV3H4600001E") -> CameraConfig:
    return CameraConfig(
        device_id=device_id,
        name=device_id,
        model="Gemini 335",
        serial=serial,
        resolution="848x480",
        fps=30,
    )


class TestParseResolution:
    def test_standard_resolution(self) -> None:
        assert _parse_resolution("848x480") == (848, 480)

    def test_uppercase_separator(self) -> None:
        assert _parse_resolution("1280X720") == (1280, 720)

    def test_extra_segments_ignored(self) -> None:
        assert _parse_resolution("640x480x30") == (640, 480)

    def test_invalid_falls_back_to_default(self) -> None:
        assert _parse_resolution("garbage") == (848, 480)


class TestStreamUrl:
    def test_serial_builds_stream_path(self) -> None:
        url = _stream_url("http://127.0.0.1:8000", _camera(serial="ABC123"), {})
        assert url == "http://127.0.0.1:8000/stream/ABC123"

    def test_trailing_slash_is_stripped(self) -> None:
        url = _stream_url("http://host:8000/", _camera(serial="ABC123"), {})
        assert url == "http://host:8000/stream/ABC123"

    def test_device_id_override_wins(self) -> None:
        overrides = {"cam_high": "http://10.0.0.5:9000/stream/custom"}
        url = _stream_url("http://127.0.0.1:8000", _camera(), overrides)
        assert url == "http://10.0.0.5:9000/stream/custom"

    def test_missing_serial_falls_back_to_device_id(self) -> None:
        url = _stream_url("http://host:8000", _camera(serial=""), {})
        assert url == "http://host:8000/stream/cam_high"


class TestCameraManagerSource:
    def test_default_source_uses_orbbec_cameras(self) -> None:
        manager = CameraManager({"cam_high": _camera()})
        assert isinstance(manager.cameras["cam_high"], OrbbecCamera)
        assert manager.device_ids() == ["cam_high"]

    def test_stream_source_uses_mjpeg_cameras(self) -> None:
        manager = CameraManager(
            {"cam_high": _camera(serial="ABC123")},
            {"source": "stream", "stream_base_url": "http://host:8000"},
        )
        cam = manager.cameras["cam_high"]
        assert isinstance(cam, MjpegCamera)
        assert cam.url == "http://host:8000/stream/ABC123"

    def test_stream_source_honors_url_override(self) -> None:
        manager = CameraManager(
            {"cam_high": _camera()},
            {
                "source": "stream",
                "stream_urls": {"cam_high": "http://override:8000/stream/x"},
            },
        )
        assert manager.cameras["cam_high"].url == "http://override:8000/stream/x"
