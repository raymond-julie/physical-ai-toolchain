"""Tests for the driver registry: register, lookup, and unknown-name errors."""

from __future__ import annotations

import pytest
from episode_recorder.drivers.base import (
    GripperDriver,
    GripperState,
    RobotState,
    RobotStateDriver,
)
from episode_recorder.drivers.noop import NoGripperDriver
from episode_recorder.drivers.registry import (
    UnknownDriverError,
    create_gripper_driver,
    create_state_driver,
    list_gripper_drivers,
    list_state_drivers,
    register_gripper_driver,
    register_state_driver,
)
from episode_recorder.drivers.ur_rtde import UrRtdeDriver


class _DummyStateDriver(RobotStateDriver):
    def __init__(self, **config: object) -> None:
        self.config = config

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    @property
    def is_connected(self) -> bool:
        return True

    def read_state(self) -> RobotState | None:
        return None


class _DummyGripperDriver(GripperDriver):
    def __init__(self, **config: object) -> None:
        self.config = config

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    @property
    def is_connected(self) -> bool:
        return True

    def read_state(self) -> GripperState | None:
        return None


class TestBundledDriversRegistered:
    def test_state_drivers_autoload(self) -> None:
        names = list_state_drivers()
        assert "ur_rtde" in names
        assert "nova" in names

    def test_gripper_drivers_autoload(self) -> None:
        names = list_gripper_drivers()
        assert "robotiq_socket" in names
        assert "none" in names
        assert "noop" in names


class TestCreateBundled:
    def test_create_state_driver_returns_instance(self) -> None:
        drv = create_state_driver("ur_rtde", host="192.168.1.80")
        assert isinstance(drv, UrRtdeDriver)
        assert drv.host == "192.168.1.80"

    def test_create_gripper_driver_returns_instance(self) -> None:
        assert isinstance(create_gripper_driver("none"), NoGripperDriver)


class TestUnknownDriver:
    def test_unknown_state_driver_raises(self) -> None:
        with pytest.raises(UnknownDriverError):
            create_state_driver("does_not_exist")

    def test_unknown_gripper_driver_raises(self) -> None:
        with pytest.raises(UnknownDriverError):
            create_gripper_driver("does_not_exist")

    def test_error_message_names_driver(self) -> None:
        with pytest.raises(UnknownDriverError, match="bogus"):
            create_state_driver("bogus")

    def test_error_is_lookup_error(self) -> None:
        # Broad ``except Exception`` callers still catch it.
        assert issubclass(UnknownDriverError, LookupError)


class TestCustomRegistration:
    def test_register_and_create_state(self) -> None:
        register_state_driver("_dummy_state", _DummyStateDriver)
        drv = create_state_driver("_dummy_state", host="x", extra=7)
        assert isinstance(drv, _DummyStateDriver)
        assert drv.config == {"host": "x", "extra": 7}
        assert "_dummy_state" in list_state_drivers()

    def test_register_and_create_gripper(self) -> None:
        register_gripper_driver("_dummy_gripper", _DummyGripperDriver)
        assert isinstance(create_gripper_driver("_dummy_gripper"), _DummyGripperDriver)
        assert "_dummy_gripper" in list_gripper_drivers()
