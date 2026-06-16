"""Driver layer — vendor-specific code lives here.

A driver is anything that implements one of the abstract bases from
:mod:`episode_recorder.drivers.base`. New robot vendors / new grippers
are added by writing a new module under this package and calling
``register_state_driver`` / ``register_gripper_driver``.

All bundled drivers are auto-imported on first call to
``create_state_driver`` / ``create_gripper_driver``.
"""

from __future__ import annotations

from .base import GripperDriver, GripperState, RobotState, RobotStateDriver
from .registry import (
    UnknownDriverError,
    create_gripper_driver,
    create_state_driver,
    register_gripper_driver,
    register_state_driver,
)

__all__ = [
    "GripperDriver",
    "GripperState",
    "RobotState",
    "RobotStateDriver",
    "UnknownDriverError",
    "create_gripper_driver",
    "create_state_driver",
    "register_gripper_driver",
    "register_state_driver",
]
