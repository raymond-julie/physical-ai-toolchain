"""Universal Robots RTDE interfaces for the leader and follower arms.

``LeaderInterface`` is read-only: joint setpoints (``getTargetQ``), tool digital
inputs, and the analog squeeze sensor that drives the gripper.

``FollowerInterface`` commands the follower via ``servoJ`` (real-time position
streaming) plus a slow ``moveJ`` used once at startup to align the follower to
the leader before live mirroring begins.

Tool analog inputs (UR pendant "tool analog input 2/3") are not exposed by every
``ur_rtde`` build. ``LeaderInterface`` resolves the analog read at connect time:
it first tries the matching ``ur_rtde`` getter, and if that is unavailable it
falls back to the official ``rtde`` client with a custom output recipe. Either
backend yields the same float reading, so the rest of the app is agnostic.

This module is part of the shelved teleop path and is not imported in
recording-only mode; it remains for when mirroring is re-enabled.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from .analog import ANALOG_INPUT_GETTERS, ANALOG_INPUT_RECIPE_FIELDS

if TYPE_CHECKING:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

_LOGGER = logging.getLogger(__name__)

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    RTDE_AVAILABLE = True
except ImportError:  # pragma: no cover - hardware dependency
    RTDE_AVAILABLE = False


# Tool digital inputs map to bits 16/17 of the digital-input register on UR.
TOOL_DI0_BIT = 16
TOOL_DI1_BIT = 17


class LeaderInterface:
    """Read-only RTDE view of a leader arm + its tool analog sensor."""

    def __init__(self, host: str, analog_input: str = "tool0") -> None:
        self.host = host
        self.analog_input = analog_input
        self._recv: RTDEReceiveInterface | None = None
        self._connected = False
        self._analog_reader: Callable[[], float] | None = None
        self._recipe_client = None  # official rtde fallback

    def connect(self) -> bool:
        if not RTDE_AVAILABLE:
            _LOGGER.error("ur_rtde not installed; cannot connect leader %s", self.host)
            return False
        try:
            self._recv = RTDEReceiveInterface(self.host)
            self._connected = True
        except Exception as exc:
            _LOGGER.error("Leader RTDE connect failed (%s): %s", self.host, exc)
            self._recv = None
            self._connected = False
            return False
        self._resolve_analog_reader()
        return True

    def disconnect(self) -> None:
        if self._recv is not None:
            with contextlib.suppress(Exception):
                self._recv.disconnect()
            self._recv = None
        if self._recipe_client is not None:
            with contextlib.suppress(Exception):
                self._recipe_client.disconnect()
            self._recipe_client = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        if not self._connected or self._recv is None:
            return False
        try:
            return bool(self._recv.isConnected())
        except Exception:
            self._connected = False
            return False

    def _resolve_analog_reader(self) -> None:
        """Pick the best available backend for the configured analog input."""
        key = self.analog_input.lower()
        getters = ANALOG_INPUT_GETTERS.get(key, ())
        for getter_name in getters:
            method = getattr(self._recv, getter_name, None)
            if callable(method):
                try:
                    method()  # probe — raises if the field is unavailable
                except Exception:
                    continue
                self._analog_reader = method
                _LOGGER.info(
                    "Leader %s analog '%s' via ur_rtde.%s", self.host, key, getter_name
                )
                return
        # Fall back to the official rtde recipe client.
        if self._setup_recipe_reader(key):
            return
        _LOGGER.warning(
            "Leader %s: no analog reader for input '%s'; gripper will hold open. "
            "Install a ur_rtde build exposing %s, or the official 'rtde' client.",
            self.host,
            key,
            getters or "tool analog getters",
        )
        self._analog_reader = lambda: 0.0

    def _setup_recipe_reader(self, key: str) -> bool:
        field = ANALOG_INPUT_RECIPE_FIELDS.get(key)
        if not field:
            return False
        try:
            import rtde.rtde as rtde_mod
        except ImportError:
            return False
        try:
            client = rtde_mod.RTDE(self.host, 30004)
            client.connect()
            client.get_controller_version()
            recipe = client.send_output_setup([field], ["DOUBLE"], frequency=125)
            if recipe is None or not client.send_start():
                client.disconnect()
                return False
        except Exception as exc:
            _LOGGER.error("Official RTDE recipe setup failed (%s): %s", self.host, exc)
            return False

        self._recipe_client = client

        def _read() -> float:
            try:
                state = client.receive()
                if state is None:
                    return 0.0
                return float(getattr(state, field))
            except Exception:
                return 0.0

        self._analog_reader = _read
        _LOGGER.info(
            "Leader %s analog '%s' via official rtde recipe (%s)", self.host, key, field
        )
        return True

    def read_target_q(self) -> list[float] | None:
        """Smooth controller setpoint; falls back to actual encoder values."""
        if not self.is_connected:
            return None
        try:
            return list(self._recv.getTargetQ())
        except Exception:
            try:
                return list(self._recv.getActualQ())
            except Exception:
                self._connected = False
                return None

    def read_tool_digital_inputs(self) -> tuple[bool, bool]:
        """Return (di0, di1) booleans for the tool digital inputs."""
        if not self.is_connected:
            return (False, False)
        try:
            bits = self._recv.getActualDigitalInputBits()
            return (
                bool(bits & (1 << TOOL_DI0_BIT)),
                bool(bits & (1 << TOOL_DI1_BIT)),
            )
        except Exception:
            self._connected = False
            return (False, False)

    def read_analog(self) -> float:
        """Raw analog reading from the configured tool sensor."""
        if self._analog_reader is None:
            return 0.0
        try:
            return float(self._analog_reader())
        except Exception:
            return 0.0


class FollowerInterface:
    """RTDE control of a follower arm via servoJ + slow moveJ alignment."""

    def __init__(self, host: str) -> None:
        self.host = host
        self._control: RTDEControlInterface | None = None
        self._recv: RTDEReceiveInterface | None = None
        self._connected = False

    def connect(self) -> bool:
        if not RTDE_AVAILABLE:
            _LOGGER.error(
                "ur_rtde not installed; cannot control follower %s", self.host
            )
            return False
        try:
            self._control = RTDEControlInterface(self.host)
            self._recv = RTDEReceiveInterface(self.host)
            self._connected = True
        except Exception as exc:
            _LOGGER.error("Follower RTDE connect failed (%s): %s", self.host, exc)
            self._connected = False
            return False
        return True

    def disconnect(self) -> None:
        if self._control is not None:
            with contextlib.suppress(Exception):
                self._control.servoStop()
            with contextlib.suppress(Exception):
                self._control.stopScript()
            with contextlib.suppress(Exception):
                self._control.disconnect()
            self._control = None
        if self._recv is not None:
            with contextlib.suppress(Exception):
                self._recv.disconnect()
            self._recv = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        if not self._connected or self._control is None:
            return False
        try:
            return bool(self._control.isConnected())
        except Exception:
            self._connected = False
            return False

    def get_actual_q(self) -> list[float] | None:
        if self._recv is None:
            return None
        try:
            return list(self._recv.getActualQ())
        except Exception:
            return None

    def move_j(
        self, q: list[float], speed: float = 0.25, acceleration: float = 0.5
    ) -> bool:
        """Blocking slow joint move — used only for startup alignment."""
        if not self.is_connected:
            return False
        try:
            return bool(self._control.moveJ(list(q), speed, acceleration))
        except Exception as exc:
            _LOGGER.error("Follower %s moveJ failed: %s", self.host, exc)
            return False

    def servo_j(
        self,
        q: list[float],
        velocity: float,
        acceleration: float,
        time_s: float,
        lookahead: float,
        gain: int,
    ) -> bool:
        """Stream one real-time servo target. Call at a fixed high rate."""
        if not self.is_connected:
            return False
        try:
            self._control.servoJ(list(q), velocity, acceleration, time_s, lookahead, gain)
            return True
        except Exception as exc:
            _LOGGER.error("Follower %s servoJ failed: %s", self.host, exc)
            self._connected = False
            return False

    def servo_stop(self) -> None:
        if self._control is not None:
            with contextlib.suppress(Exception):
                self._control.servoStop()
