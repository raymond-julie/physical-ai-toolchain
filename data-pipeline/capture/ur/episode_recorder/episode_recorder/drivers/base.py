"""Abstract base classes for robot-state and gripper drivers.

The recorder is intentionally decoupled from any specific robot vendor.
To support a new robot, write a subclass of :class:`RobotStateDriver`
and register it with :func:`register_state_driver`. Likewise for
grippers via :class:`GripperDriver`.

Drivers are read-only by design — the recorder app never commands the
robot. The optional :meth:`RobotStateDriver.set_digital_output` exists
only to drive a visible "recording" LED (e.g. UR tool DO0) when the
hardware supports it. Implementations that cannot drive an output
should leave the default no-op.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RobotState:
    """Read-only snapshot of a robot's instantaneous state.

    Attributes:
        joint_names: Joint names in publication order (length == DOF).
        joint_positions: Joint positions, same order as ``joint_names``.
        joint_velocities: Joint velocities, same order. May be all zeros
            if the driver cannot read velocities.
        digital_inputs: Optional mapping of human-readable name -> value
            for any boolean inputs the driver wants to expose (e.g.
            ``{"di0": True, "di1": False}``). The keys become ROS topic
            suffixes under ``<robot_name>/digital_input/<key>``.
    """

    joint_names: list[str]
    joint_positions: list[float]
    joint_velocities: list[float]
    digital_inputs: dict[str, bool] = field(default_factory=dict)


@dataclass
class GripperState:
    """Read-only snapshot of a gripper's instantaneous state.

    Attributes:
        position: Normalized opening in [0.0, 1.0]; 0.0 = fully open,
            1.0 = fully closed.
        is_closed: Convenience boolean — typically ``position >= threshold``.
    """

    position: float
    is_closed: bool


class RobotStateDriver(ABC):
    """Vendor-agnostic interface to a single robot's state.

    Driver implementations must accept their configuration as
    keyword-only arguments to ``__init__`` so the registry can build
    them from a flat config dict. Unknown kwargs should be ignored
    (``**_kwargs``) so configs can carry shared keys safely.

    All methods are expected to be called from a single thread.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Open the underlying connection. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release any underlying resources. Idempotent."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True iff the driver currently has a live connection."""

    @abstractmethod
    def read_state(self) -> RobotState | None:
        """Return the latest state, or None on transient read failure.

        Returning None should leave ``is_connected`` accurate so callers
        can decide whether to attempt reconnection.
        """

    def set_digital_output(self, name: str, value: bool) -> bool:
        """Optionally drive a robot-side digital output.

        Default implementation returns False (not supported). Drivers
        that can drive outputs should override. ``name`` is a
        driver-specific identifier (e.g. ``"do0"`` for UR tool DO0).
        """
        return False


class GripperDriver(ABC):
    """Vendor-agnostic interface to a single gripper's state."""

    @abstractmethod
    def connect(self) -> bool:
        """Open the underlying connection. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release any underlying resources. Idempotent."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True iff the driver currently has a live connection."""

    @abstractmethod
    def read_state(self) -> GripperState | None:
        """Return the latest gripper state, or None on read failure."""
