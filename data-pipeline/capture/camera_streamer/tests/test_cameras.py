"""Tests for camera_streamer pure helpers (no hardware)."""

from __future__ import annotations

from camera_streamer.cameras import ORBBEC_AVAILABLE, _parse_resolution


class TestParseResolution:
    def test_valid(self) -> None:
        assert _parse_resolution("1280x720") == (1280, 720)

    def test_case_insensitive(self) -> None:
        assert _parse_resolution("640X480") == (640, 480)

    def test_invalid_falls_back(self) -> None:
        assert _parse_resolution("not-a-res") == (848, 480)


class TestOrbbecAvailability:
    def test_orbbec_unavailable_in_test_env(self) -> None:
        assert ORBBEC_AVAILABLE is False
