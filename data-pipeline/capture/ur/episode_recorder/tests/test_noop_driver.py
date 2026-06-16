"""Tests for the no-op gripper driver."""

from __future__ import annotations

from episode_recorder.drivers.noop import NoGripperDriver
from episode_recorder.drivers.registry import create_gripper_driver


class TestNoGripperDriver:
    def test_connect_succeeds(self) -> None:
        assert NoGripperDriver().connect() is True

    def test_is_connected_always_true(self) -> None:
        assert NoGripperDriver().is_connected is True

    def test_read_state_is_open(self) -> None:
        state = NoGripperDriver().read_state()
        assert state is not None
        assert state.position == 0.0
        assert state.is_closed is False

    def test_disconnect_is_noop(self) -> None:
        drv = NoGripperDriver()
        drv.disconnect()
        assert drv.is_connected is True

    def test_absorbs_arbitrary_kwargs(self) -> None:
        drv = NoGripperDriver(host="192.168.1.80", port=63352, closed_threshold=128)
        assert drv.connect() is True

    def test_registered_under_none_and_noop(self) -> None:
        assert isinstance(create_gripper_driver("none"), NoGripperDriver)
        assert isinstance(create_gripper_driver("noop"), NoGripperDriver)
