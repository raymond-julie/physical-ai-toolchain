"""Robotiq 2F-85 gripper client over the URCap socket protocol (port 63352).

The Robotiq URCap exposes a line-based ASCII protocol tunnelled through the UR
controller. This client supports both reading (``GET POS``) and writing
(``SET POS/SPE/FOR`` + ``SET GTO 1``), with a thin convenience layer for issuing
*position-rate-limited* commands driven by an analog sensor.

Position convention (matches Robotiq firmware):

* ``0``   = fully open
* ``255`` = fully closed
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time


class RobotiqGripper:
    """Thread-safe Robotiq 2F-85 socket client (read + write)."""

    def __init__(
        self,
        host: str,
        port: int = 63352,
        timeout: float = 1.0,
    ) -> None:
        self.host = host
        self.port = int(port) if port else 63352
        self.timeout = float(timeout)
        self._sock: socket.socket | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._last_sent_pos = -1

    def connect(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            with self._lock:
                self._sock = sock
                self._connected = True
        except OSError:
            self._sock = None
            self._connected = False
            return False
        # Activation is idempotent on an already-active gripper.
        self._send("SET ACT 1")
        self._drain()
        return True

    def disconnect(self) -> None:
        with self._lock:
            if self._sock is not None:
                with contextlib.suppress(OSError):
                    self._sock.close()
                self._sock = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _send(self, cmd: str) -> str | None:
        with self._lock:
            if not self._connected or self._sock is None:
                return None
            try:
                self._sock.sendall(f"{cmd}\n".encode())
                return self._sock.recv(1024).decode().strip()
            except OSError:
                self._connected = False
                return None

    def _drain(self) -> None:
        with self._lock:
            if self._sock is None:
                return
            try:
                self._sock.setblocking(False)
                while True:
                    try:
                        if not self._sock.recv(1024):
                            break
                    except BlockingIOError:
                        break
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    self._sock.setblocking(True)
                    self._sock.settimeout(self.timeout)

    def get_position(self) -> int:
        """Return raw position 0..255, or -1 on failure."""
        resp = self._send("GET POS")
        if resp and resp.startswith("POS"):
            try:
                return int(resp.split()[1])
            except (IndexError, ValueError):
                return -1
        return -1

    def move(self, position: int, speed: int = 255, force: int = 80) -> bool:
        """Command an absolute position (0..255) with speed/force (0..255)."""
        if not self._connected:
            return False
        pos = max(0, min(255, int(position)))
        spe = max(0, min(255, int(speed)))
        frc = max(0, min(255, int(force)))
        ok = self._send(f"SET POS {pos}") is not None
        self._send(f"SET SPE {spe}")
        self._send(f"SET FOR {frc}")
        self._send("SET GTO 1")
        if ok:
            self._last_sent_pos = pos
        return ok

    def move_if_changed(
        self,
        position: int,
        min_delta: int = 1,
        speed: int = 255,
        force: int = 80,
    ) -> bool:
        """Send a move only when ``position`` differs from the last command.

        Avoids saturating the (slow) gripper socket when an analog sensor is
        producing a continuous stream of near-identical targets.
        """
        pos = max(0, min(255, int(position)))
        if self._last_sent_pos >= 0 and abs(pos - self._last_sent_pos) < min_delta:
            return True
        return self.move(pos, speed=speed, force=force)


class RobotiqReconnector:
    """Keeps a :class:`RobotiqGripper` connected with periodic retries."""

    def __init__(self, gripper: RobotiqGripper, retry_interval: float = 3.0) -> None:
        self.gripper = gripper
        self.retry_interval = retry_interval
        self._last_attempt = 0.0

    def ensure(self) -> bool:
        if self.gripper.is_connected:
            return True
        now = time.monotonic()
        if now - self._last_attempt < self.retry_interval:
            return False
        self._last_attempt = now
        return self.gripper.connect()
