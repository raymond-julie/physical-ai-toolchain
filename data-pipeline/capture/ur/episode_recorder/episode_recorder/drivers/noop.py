"""No-op gripper driver — for robots that do not have a gripper.

Returns ``position=0.0, is_closed=False`` forever, ``connect()``
always succeeds. Register as gripper name ``"none"`` or ``"noop"``.
"""

from __future__ import annotations

from typing import Any

from .base import GripperDriver, GripperState
from .registry import register_gripper_driver


class NoGripperDriver(GripperDriver):
    """Stub driver used when a robot has no gripper hardware."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    def read_state(self) -> GripperState | None:
        return GripperState(position=0.0, is_closed=False)


register_gripper_driver("none", NoGripperDriver)
register_gripper_driver("noop", NoGripperDriver)
