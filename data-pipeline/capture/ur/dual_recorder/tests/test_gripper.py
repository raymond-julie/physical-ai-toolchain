"""Tests for ur_dual_recorder gripper logic (no robot, no socket).

Covers the closed-threshold derivation in :class:`ArmReader` (raw position ->
normalized position + closed flag) and the pure parts of the Robotiq socket
client (command clamping, position parsing, change suppression, reconnect
gating).
"""

from __future__ import annotations

import pytest
from ur_dual_recorder.arm_reader import ArmReader
from ur_dual_recorder.robotiq import RobotiqGripper, RobotiqReconnector

_TICK_NOW = 1_000_000_000.0  # well past the gripper throttle window


class TestArmReaderThresholdConfig:
    def test_default_threshold_is_128(self) -> None:
        reader = ArmReader("left", "192.168.1.11", {})
        assert reader.gripper_closed_threshold == 128

    def test_threshold_and_port_from_config(self) -> None:
        reader = ArmReader("left", "192.168.1.11", {"closed_threshold": 200, "port": 12345})
        assert reader.gripper_closed_threshold == 200
        assert reader.gripper.port == 12345


class TestArmReaderClosedLogic:
    def _reader(self, monkeypatch: pytest.MonkeyPatch, threshold: int, raw: int) -> ArmReader:
        reader = ArmReader("left", "192.168.1.11", {"closed_threshold": threshold})
        monkeypatch.setattr(reader._gripper_recon, "ensure", lambda: True)
        monkeypatch.setattr(reader.gripper, "get_position", lambda: raw)
        return reader

    def test_raw_above_threshold_is_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reader = self._reader(monkeypatch, threshold=128, raw=200)
        reader._tick(_TICK_NOW)
        sample = reader.sample()
        assert sample.gripper_is_closed is True
        assert sample.gripper_position == pytest.approx(200 / 255.0)

    def test_raw_below_threshold_is_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reader = self._reader(monkeypatch, threshold=128, raw=100)
        reader._tick(_TICK_NOW)
        sample = reader.sample()
        assert sample.gripper_is_closed is False
        assert sample.gripper_position == pytest.approx(100 / 255.0)

    def test_raw_equal_to_threshold_is_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The closed test is strictly greater-than, so equality stays open.
        reader = self._reader(monkeypatch, threshold=128, raw=128)
        reader._tick(_TICK_NOW)
        assert reader.sample().gripper_is_closed is False

    def test_failed_read_leaves_sample_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reader = self._reader(monkeypatch, threshold=128, raw=-1)
        reader._tick(_TICK_NOW)
        sample = reader.sample()
        assert sample.gripper_position == 0.0
        assert sample.gripper_is_closed is False


class TestRobotiqGripperCommands:
    def _connected_gripper(self, monkeypatch: pytest.MonkeyPatch) -> RobotiqGripper:
        gripper = RobotiqGripper("192.168.1.11")
        gripper._connected = True
        monkeypatch.setattr(gripper, "_send", lambda cmd: "ack")
        return gripper

    def test_move_clamps_to_valid_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gripper = self._connected_gripper(monkeypatch)
        assert gripper.move(300) is True
        assert gripper._last_sent_pos == 255
        assert gripper.move(-5) is True
        assert gripper._last_sent_pos == 0

    def test_move_when_disconnected_returns_false(self) -> None:
        gripper = RobotiqGripper("192.168.1.11")
        assert gripper.move(100) is False

    def test_move_if_changed_suppresses_small_deltas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gripper = self._connected_gripper(monkeypatch)
        gripper._last_sent_pos = 100
        calls: list[int] = []
        monkeypatch.setattr(gripper, "move", lambda pos, **k: calls.append(pos) or True)
        assert gripper.move_if_changed(100, min_delta=1) is True
        assert calls == []  # within deadband -> no command sent

    def test_move_if_changed_sends_on_large_delta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gripper = self._connected_gripper(monkeypatch)
        gripper._last_sent_pos = 100
        calls: list[int] = []
        monkeypatch.setattr(gripper, "move", lambda pos, **k: calls.append(pos) or True)
        gripper.move_if_changed(150)
        assert calls == [150]


class TestRobotiqGripperGetPosition:
    def _gripper(self, monkeypatch: pytest.MonkeyPatch, response: str | None) -> RobotiqGripper:
        gripper = RobotiqGripper("192.168.1.11")
        gripper._connected = True
        monkeypatch.setattr(gripper, "_send", lambda cmd: response)
        return gripper

    def test_valid_response_is_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._gripper(monkeypatch, "POS 128").get_position() == 128

    def test_non_numeric_value_returns_minus_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._gripper(monkeypatch, "POS abc").get_position() == -1

    def test_missing_value_returns_minus_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._gripper(monkeypatch, "POS").get_position() == -1

    def test_unexpected_prefix_returns_minus_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._gripper(monkeypatch, "ACK 1").get_position() == -1

    def test_no_response_returns_minus_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._gripper(monkeypatch, None).get_position() == -1


class TestRobotiqReconnector:
    def test_already_connected_skips_connect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gripper = RobotiqGripper("192.168.1.11")
        gripper._connected = True
        calls: list[bool] = []
        monkeypatch.setattr(gripper, "connect", lambda: calls.append(True) or True)
        assert RobotiqReconnector(gripper).ensure() is True
        assert calls == []

    def test_retry_is_gated_by_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gripper = RobotiqGripper("192.168.1.11")
        recon = RobotiqReconnector(gripper, retry_interval=3.0)
        monkeypatch.setattr("ur_dual_recorder.robotiq.time.monotonic", lambda: 100.0)
        recon._last_attempt = 99.0  # only 1s ago, interval is 3s
        assert recon.ensure() is False

    def test_reconnects_after_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gripper = RobotiqGripper("192.168.1.11")
        recon = RobotiqReconnector(gripper, retry_interval=3.0)
        monkeypatch.setattr("ur_dual_recorder.robotiq.time.monotonic", lambda: 100.0)
        recon._last_attempt = 90.0  # well past the retry interval
        monkeypatch.setattr(gripper, "connect", lambda: True)
        assert recon.ensure() is True
