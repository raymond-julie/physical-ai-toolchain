"""Driver registry — string name -> class lookup.

Drivers register themselves by calling :func:`register_state_driver`
or :func:`register_gripper_driver` at module import time. The first
call to :func:`create_state_driver` / :func:`create_gripper_driver`
auto-imports all bundled drivers so they self-register.
"""

from __future__ import annotations

from typing import Any

from .base import GripperDriver, RobotStateDriver

_STATE_DRIVERS: dict[str, type[RobotStateDriver]] = {}
_GRIPPER_DRIVERS: dict[str, type[GripperDriver]] = {}
_AUTOLOADED = False


class UnknownDriverError(LookupError):
    """Raised when a driver name is not present in the registry."""


def register_state_driver(name: str, cls: type[RobotStateDriver]) -> None:
    """Register a robot-state driver class under a short name."""
    _STATE_DRIVERS[name] = cls


def register_gripper_driver(name: str, cls: type[GripperDriver]) -> None:
    """Register a gripper driver class under a short name."""
    _GRIPPER_DRIVERS[name] = cls


def list_state_drivers() -> list[str]:
    """Return the sorted names of every registered state driver."""
    _autoload()
    return sorted(_STATE_DRIVERS)


def list_gripper_drivers() -> list[str]:
    """Return the sorted names of every registered gripper driver."""
    _autoload()
    return sorted(_GRIPPER_DRIVERS)


def create_state_driver(name: str, **config: Any) -> RobotStateDriver:
    """Instantiate a registered state driver by name."""
    _autoload()
    if name not in _STATE_DRIVERS:
        raise UnknownDriverError(
            f"Unknown state driver '{name}'. Known: {sorted(_STATE_DRIVERS)}"
        )
    return _STATE_DRIVERS[name](**config)


def create_gripper_driver(name: str, **config: Any) -> GripperDriver:
    """Instantiate a registered gripper driver by name."""
    _autoload()
    if name not in _GRIPPER_DRIVERS:
        raise UnknownDriverError(
            f"Unknown gripper driver '{name}'. Known: {sorted(_GRIPPER_DRIVERS)}"
        )
    return _GRIPPER_DRIVERS[name](**config)


def _autoload() -> None:
    """Import bundled driver modules once so they self-register."""
    global _AUTOLOADED
    if _AUTOLOADED:
        return
    _AUTOLOADED = True
    from . import noop, nova, robotiq, ur_rtde  # noqa: F401
