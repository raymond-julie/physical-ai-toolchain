"""Universal Robots + Robotiq state driver via ur_rtde (read-only).

Maps UR's ``getActualDigitalInputBits`` bits 16/17 (tool DI0/DI1) into
the generic ``RobotState.digital_inputs`` dict as ``"di0"`` / ``"di1"``.
The mapping is overridable via the ``digital_input_names`` constructor
kwarg so the same driver can be reused for non-tool inputs.

The driver also implements :meth:`set_digital_output` for UR tool
DOs (e.g. ``"do0"`` -> ``setToolDigitalOut(0, ...)``) so a "recording"
LED can be driven on the robot tool without leaving the abstract API.
"""

from __future__ import annotations

import contextlib
from typing import Any

from .base import RobotState, RobotStateDriver
from .registry import register_state_driver

try:
    from rtde_io import RTDEIOInterface
    from rtde_receive import RTDEReceiveInterface

    _RTDE_AVAILABLE = True
except ImportError:
    _RTDE_AVAILABLE = False


# Default UR joint name list (matches UR5e/UR10e/UR3e).
UR_JOINT_NAMES: list[str] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class UrRtdeDriver(RobotStateDriver):
    """Read-only UR + RTDE state driver.

    Args:
        host: Robot controller IP address.
        port: Ignored (RTDE picks its own ports). Accepted for
            registry-config uniformity.
        joint_names: Override the default UR joint name list.
        digital_input_names: Mapping ``{bit_index: name}``. Default
            ``{16: "di0", 17: "di1"}`` matches UR tool digital inputs.
        prefer_target_q: If True (default), publish controller setpoint
            (``getTargetQ``) — smoother for downstream consumers. Falls
            back to ``getActualQ`` if the firmware lacks the call.
    """

    def __init__(
        self,
        host: str,
        port: int = 0,
        joint_names: list[str] | None = None,
        digital_input_names: dict[int, str] | None = None,
        prefer_target_q: bool = True,
        **_kwargs: Any,
    ) -> None:
        self.host = host
        self.joint_names = list(joint_names) if joint_names else UR_JOINT_NAMES
        self._di_map: dict[int, str] = (
            dict(digital_input_names) if digital_input_names else {16: "di0", 17: "di1"}
        )
        self._prefer_target = bool(prefer_target_q)
        self._rtde = None
        self._io = None
        self._connected = False

    def connect(self) -> bool:
        if not _RTDE_AVAILABLE:
            return False
        try:
            self._rtde = RTDEReceiveInterface(self.host)
            self._connected = True
        except Exception:
            self._rtde = None
            self._connected = False
            return False
        # IO is optional — needed only to drive the recording LED.
        try:
            self._io = RTDEIOInterface(self.host)
        except Exception:
            self._io = None
        return self._connected

    def disconnect(self) -> None:
        if self._rtde is not None:
            with contextlib.suppress(Exception):
                self._rtde.disconnect()
            self._rtde = None
        if self._io is not None:
            with contextlib.suppress(Exception):
                self._io.disconnect()
            self._io = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        if not self._connected or self._rtde is None:
            return False
        try:
            return bool(self._rtde.isConnected())
        except Exception:
            self._connected = False
            return False

    def read_state(self) -> RobotState | None:
        if not self.is_connected:
            return None
        try:
            if self._prefer_target:
                try:
                    pos = list(self._rtde.getTargetQ())
                except Exception:
                    pos = list(self._rtde.getActualQ())
                try:
                    vel = list(self._rtde.getTargetQd())
                except Exception:
                    vel = list(self._rtde.getActualQd())
            else:
                pos = list(self._rtde.getActualQ())
                vel = list(self._rtde.getActualQd())
            bits = self._rtde.getActualDigitalInputBits()
            dis = {name: bool(bits & (1 << bit)) for bit, name in self._di_map.items()}
            return RobotState(
                joint_names=self.joint_names,
                joint_positions=pos,
                joint_velocities=vel,
                digital_inputs=dis,
            )
        except Exception:
            self._connected = False
            return None

    def set_digital_output(self, name: str, value: bool) -> bool:
        """Set a UR tool digital output. ``name`` must be ``"doN"``."""
        if self._io is None:
            return False
        if not name.startswith("do"):
            return False
        try:
            n = int(name[2:])
        except ValueError:
            return False
        try:
            self._io.setToolDigitalOut(n, bool(value))
            return True
        except Exception:
            return False


register_state_driver("ur_rtde", UrRtdeDriver)
