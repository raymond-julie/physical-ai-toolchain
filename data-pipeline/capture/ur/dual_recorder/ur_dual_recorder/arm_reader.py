"""Read-only state reader for a single UR arm + its Robotiq gripper.

Used by the recording-only pipeline: the follower arms are **observed**, never
commanded. Each :class:`ArmReader` owns a background thread that keeps the latest
joint positions, gripper position/closed-state, and tool DI0 button cached so the
recorder's frame provider never blocks on a socket round-trip.

No ``RTDEControlInterface`` is created, so this reader never takes the RTDE
control lock and can safely coexist with anything else driving the arm.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .robotiq import RobotiqGripper, RobotiqReconnector

if TYPE_CHECKING:
    from rtde_receive import RTDEReceiveInterface

_LOGGER = logging.getLogger(__name__)

try:
    from rtde_receive import RTDEReceiveInterface

    RTDE_AVAILABLE = True
except ImportError:  # pragma: no cover - hardware dependency
    RTDE_AVAILABLE = False


TOOL_DI0_BIT = 16


@dataclass
class ArmSample:
    """Latest observed state of one arm."""

    joints: list[float] = field(default_factory=lambda: [0.0] * 6)
    gripper_position: float = 0.0  # normalized 0..1 (1 = closed)
    gripper_is_closed: bool = False
    di0: bool = False
    arm_connected: bool = False
    gripper_connected: bool = False


class ArmReader:
    """Background read-only reader for one arm's joints + gripper + DI0."""

    def __init__(
        self,
        name: str,
        ip: str,
        gripper_cfg: dict,
        joint_rate: float = 60.0,
        gripper_rate: float = 10.0,
        reconnect_interval: float = 3.0,
    ) -> None:
        self.name = name
        self.ip = ip
        self.gripper_closed_threshold = int(gripper_cfg.get("closed_threshold", 128))
        self.gripper = RobotiqGripper(ip, port=int(gripper_cfg.get("port", 63352)))
        self._gripper_recon = RobotiqReconnector(self.gripper, reconnect_interval)

        self._recv: RTDEReceiveInterface | None = None
        self._joint_period = 1.0 / max(1.0, joint_rate)
        self._gripper_period = 1.0 / max(1.0, gripper_rate)
        self._reconnect_interval = reconnect_interval

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sample = ArmSample()
        self._last_gripper_t = 0.0
        self._last_rtde_attempt = 0.0

        # DI0 rising-edge trigger (optional record button on the tool).
        self._di0_prev = False
        self.di0_edge = threading.Event()

    def _connect_rtde(self) -> bool:
        if not RTDE_AVAILABLE:
            return False
        now = time.monotonic()
        if now - self._last_rtde_attempt < self._reconnect_interval:
            return False
        self._last_rtde_attempt = now
        try:
            self._recv = RTDEReceiveInterface(self.ip)
            _LOGGER.info("[%s] RTDE receive connected (%s)", self.name, self.ip)
            return True
        except Exception as exc:
            _LOGGER.error("[%s] RTDE connect failed (%s): %s", self.name, self.ip, exc)
            self._recv = None
            return False

    def _rtde_ok(self) -> bool:
        if self._recv is None:
            return False
        try:
            return bool(self._recv.isConnected())
        except Exception:
            return False

    def connect(self) -> None:
        self._connect_rtde()
        self.gripper.connect()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"arm-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def shutdown(self) -> None:
        self.stop()
        if self._recv is not None:
            with contextlib.suppress(Exception):
                self._recv.disconnect()
            self._recv = None
        self.gripper.disconnect()

    def _run(self) -> None:
        _LOGGER.info("[%s] reader started", self.name)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._tick(t0)
            except Exception as exc:
                _LOGGER.debug("[%s] tick error: %s", self.name, exc)
            sleep = self._joint_period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _tick(self, now: float) -> None:
        # Joints + DI0 (fast, local RTDE read).
        joints = None
        di0 = False
        if self._rtde_ok():
            try:
                joints = list(self._recv.getActualQ())
                bits = self._recv.getActualDigitalInputBits()
                di0 = bool(bits & (1 << TOOL_DI0_BIT))
            except Exception:
                self._recv = None
        else:
            self._connect_rtde()

        if di0 and not self._di0_prev:
            self.di0_edge.set()
        self._di0_prev = di0

        # Gripper (slower socket round-trip; throttled).
        gripper_pos = None
        gripper_closed = False
        if (now - self._last_gripper_t) >= self._gripper_period:
            self._last_gripper_t = now
            if self._gripper_recon.ensure():
                raw = self.gripper.get_position()
                if raw >= 0:
                    gripper_pos = raw / 255.0
                    gripper_closed = raw > self.gripper_closed_threshold

        with self._lock:
            if joints is not None:
                self._sample.joints = joints
            if gripper_pos is not None:
                self._sample.gripper_position = gripper_pos
                self._sample.gripper_is_closed = gripper_closed
            self._sample.di0 = di0
            self._sample.arm_connected = self._recv is not None
            self._sample.gripper_connected = self.gripper.is_connected

    def sample(self) -> ArmSample:
        with self._lock:
            s = self._sample
            return ArmSample(
                joints=list(s.joints),
                gripper_position=s.gripper_position,
                gripper_is_closed=s.gripper_is_closed,
                di0=s.di0,
                arm_connected=s.arm_connected,
                gripper_connected=s.gripper_connected,
            )
