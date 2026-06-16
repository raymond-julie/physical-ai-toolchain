"""Robotiq 2F gripper read-only driver via raw socket (port 63352).

Communicates with the Robotiq URCap socket interface exposed by the
robot controller. Position is reported as 0..255; we normalize to
[0.0, 1.0] in ``GripperState.position``.
"""

from __future__ import annotations

import contextlib
import socket
from typing import Any

from .base import GripperDriver, GripperState
from .registry import register_gripper_driver


class RobotiqSocketDriver(GripperDriver):
    """Read-only Robotiq driver.

    Args:
        host: Robot controller IP (the gripper is tunnelled through it).
        port: Robotiq URCap socket port (default 63352).
        closed_threshold: Raw position (0..255) at/above which the
            gripper counts as "closed" (default 128).
        timeout: Socket timeout in seconds.
    """

    def __init__(
        self,
        host: str,
        port: int = 63352,
        closed_threshold: int = 128,
        timeout: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        self.host = host
        self.port = int(port) if port else 63352
        self.closed_threshold = int(closed_threshold)
        self.timeout = float(timeout)
        self._sock: socket.socket | None = None
        self._connected = False

    def connect(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            self._sock = sock
            self._connected = True
        except OSError:
            self._sock = None
            self._connected = False
            return False
        # Best-effort activate. Idempotent on an already-active gripper.
        self._send_command("SET ACT 1")
        return True

    def disconnect(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_state(self) -> GripperState | None:
        pos = self._get_position()
        if pos < 0:
            self._connected = False
            return None
        return GripperState(
            position=pos / 255.0,
            is_closed=pos > self.closed_threshold,
        )

    def _get_position(self) -> int:
        resp = self._send_command("GET POS")
        if resp and resp.startswith("POS"):
            try:
                return int(resp.split()[1])
            except (IndexError, ValueError):
                return -1
        return -1

    def _send_command(self, cmd: str) -> str | None:
        if not self._connected or self._sock is None:
            return None
        try:
            self._sock.sendall(f"{cmd}\n".encode())
            return self._sock.recv(1024).decode().strip()
        except OSError:
            self._connected = False
            return None


register_gripper_driver("robotiq_socket", RobotiqSocketDriver)
