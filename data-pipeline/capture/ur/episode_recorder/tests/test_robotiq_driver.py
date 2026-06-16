"""Tests for the Robotiq gripper driver normalization (no socket)."""

from __future__ import annotations

import pytest
from episode_recorder.drivers.robotiq import RobotiqSocketDriver


def _driver_with_response(response: str | None, **kwargs: object) -> RobotiqSocketDriver:
    drv = RobotiqSocketDriver(host="192.168.1.80", **kwargs)
    drv._connected = True
    drv._send_command = lambda _cmd: response
    return drv


class TestReadStateNormalization:
    def test_fully_open(self) -> None:
        state = _driver_with_response("POS 0").read_state()
        assert state is not None
        assert state.position == 0.0
        assert state.is_closed is False

    def test_fully_closed(self) -> None:
        state = _driver_with_response("POS 255").read_state()
        assert state is not None
        assert state.position == 1.0
        assert state.is_closed is True

    def test_position_scaled(self) -> None:
        state = _driver_with_response("POS 200").read_state()
        assert state is not None
        assert state.position == pytest.approx(200 / 255)
        assert state.is_closed is True

    def test_open_below_threshold(self) -> None:
        state = _driver_with_response("POS 100").read_state()
        assert state is not None
        assert state.is_closed is False

    def test_custom_threshold(self) -> None:
        state = _driver_with_response("POS 60", closed_threshold=50).read_state()
        assert state is not None
        assert state.is_closed is True


class TestReadStateFailures:
    def test_unparseable_response_returns_none(self) -> None:
        drv = _driver_with_response("ERR")
        assert drv.read_state() is None
        assert drv.is_connected is False

    def test_missing_number_returns_none(self) -> None:
        assert _driver_with_response("POS").read_state() is None

    def test_non_numeric_position_returns_none(self) -> None:
        assert _driver_with_response("POS abc").read_state() is None

    def test_none_response_returns_none(self) -> None:
        assert _driver_with_response(None).read_state() is None


class TestSendCommandGuard:
    def test_returns_none_when_not_connected(self) -> None:
        drv = RobotiqSocketDriver(host="x")
        assert drv._send_command("GET POS") is None
