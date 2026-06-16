"""Wandelbots Nova state driver (NATS-based, read-only).

Subscribes to the Nova v2 controller-state subject for one robot and
exposes the latest state through the generic :class:`RobotStateDriver`
contract.

Subject template (Nova v2)::

    nova.v2.cells.{cell}.controllers.{controller}.state

Field mapping (Nova v2 -> :class:`RobotState`)::

    motion_groups[0].joint_position  -> joint_positions
    UR_JOINT_NAMES (overridable)     -> joint_names
    [0.0] * N                        -> joint_velocities    (not exposed by Nova v2)
    {}                               -> digital_inputs       (not exposed by Nova v2)

Nova v2's state stream does not contain joint velocities, TCP wrench,
or tool DIs. The recorder still works — the ``observation.state``
tensor is well-defined — but downstream consumers that need
velocities should source them from elsewhere.

Implementation
--------------
``nats-py`` is async-only, but the driver API is synchronous. We run
the NATS client inside a daemon thread with its own asyncio loop and
publish the latest payload into a lock-guarded slot. :meth:`read_state`
reads that slot and returns ``None`` until the first message arrives.
:meth:`is_connected` reports staleness so the reader's reconnect timer
can revive a silent subscription.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from typing import Any

from .base import RobotState, RobotStateDriver
from .registry import register_state_driver
from .ur_rtde import UR_JOINT_NAMES  # reuse the canonical UR joint list

_LOGGER = logging.getLogger(__name__)

try:
    import nats  # noqa: F401

    _NATS_AVAILABLE = True
except ImportError:
    _NATS_AVAILABLE = False


# After this many seconds without a NATS message, treat the connection
# as stale so the reader's reconnect timer kicks in.
_STALE_AFTER_S = 3.0


class NovaDriver(RobotStateDriver):
    """Read-only Wandelbots Nova state driver.

    Args:
        nats_url: NATS server URL, e.g. ``"nats://192.168.1.244:31422"``.
        cell: Nova cell id (subject placeholder ``{cell}``).
        controller: Nova controller id (subject placeholder ``{controller}``),
            e.g. ``"ur5-left"``.
        nats_user / nats_password: optional username/password auth.
        nats_creds_file: optional JWT credentials file (alternative).
        subject_template: subject pattern; defaults to the Nova v2 layout.
        joint_names: override the default UR joint name list.

    The constructor accepts ``**_kwargs`` so the same flat config dict
    used by other drivers (``host``, ``port``, ...) is silently absorbed.
    """

    def __init__(
        self,
        nats_url: str = "nats://localhost:4222",
        cell: str = "cell",
        controller: str = "ur5",
        nats_user: str = "",
        nats_password: str = "",
        nats_creds_file: str = "",
        subject_template: str = "nova.v2.cells.{cell}.controllers.{controller}.state",
        joint_names: list[str] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.nats_url = str(nats_url)
        self.cell = str(cell)
        self.controller = str(controller)
        self.nats_user = str(nats_user or "")
        self.nats_password = str(nats_password or "")
        self.nats_creds = str(nats_creds_file or "")
        self.subject = subject_template.format(cell=self.cell, controller=self.controller)
        self.joint_names: list[str] = list(joint_names) if joint_names else list(UR_JOINT_NAMES)

        # Thread-safe slot for the latest payload.
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._last_update: float = 0.0  # monotonic seconds

        # Background thread + loop handles.
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    # ── RobotStateDriver API ────────────────────────────────────

    def connect(self) -> bool:
        """Start the background NATS client.

        Returns True if the worker thread is running (the actual NATS
        handshake completes asynchronously). Subsequent :meth:`is_connected`
        calls report whether a fresh message has been received.

        Idempotent: a no-op if the worker is already alive.
        """
        if not _NATS_AVAILABLE:
            _LOGGER.error(
                "NovaDriver[%s]: 'nats-py' is not installed. Install it "
                "with: pip install --user nats-py",
                self.controller,
            )
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        _LOGGER.info(
            "NovaDriver[%s]: starting NATS worker url=%s subject=%s",
            self.controller,
            self.nats_url,
            self.subject,
        )

        # Reset slot for a fresh attempt.
        with self._lock:
            self._latest = None
            self._last_update = 0.0
        self._loop = None
        self._stop_event = None

        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"NovaDriver[{self.controller}]",
            daemon=True,
        )
        self._thread.start()
        return True

    def disconnect(self) -> None:
        loop = self._loop
        stop = self._stop_event
        if loop is not None and stop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._loop = None
        self._stop_event = None
        with self._lock:
            self._latest = None
            self._last_update = 0.0

    @property
    def is_connected(self) -> bool:
        if self._thread is None or not self._thread.is_alive():
            return False
        with self._lock:
            if self._last_update == 0.0:
                return False  # subscribed but no message yet
            age = time.monotonic() - self._last_update
        return age < _STALE_AFTER_S

    def read_state(self) -> RobotState | None:
        with self._lock:
            state = self._latest
        if state is None:
            return None
        mgs = state.get("motion_groups") or []
        mg = mgs[0] if mgs else {}
        joint_pos = mg.get("joint_position") or []
        if not joint_pos:
            return None

        # If Nova reports more/fewer joints than the configured name
        # list, adjust the names to match — better than dropping the data.
        n = len(joint_pos)
        if n == len(self.joint_names):
            names = self.joint_names
        elif n < len(self.joint_names):
            names = self.joint_names[:n]
        else:
            names = self.joint_names + [f"joint_{i}" for i in range(len(self.joint_names), n)]

        return RobotState(
            joint_names=list(names),
            joint_positions=[float(x) for x in joint_pos],
            joint_velocities=[0.0] * n,
            digital_inputs={},
        )

    # ── Background asyncio loop ─────────────────────────────────

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            self._stop_event = asyncio.Event()
            with contextlib.suppress(Exception):
                # Thread exits on error; reader's reconnect timer retries.
                loop.run_until_complete(self._serve())
        finally:
            with contextlib.suppress(Exception):
                loop.close()

    async def _serve(self) -> None:
        import nats

        opts: dict[str, Any] = {
            "servers": [self.nats_url],
            "name": f"episode_recorder.{self.controller}",
            # Bounded reconnect attempts so a permanently-down NATS
            # eventually exits the thread and lets connect() retry.
            "max_reconnect_attempts": 5,
            "reconnect_time_wait": 1.0,
        }
        if self.nats_creds:
            opts["user_credentials"] = self.nats_creds
        elif self.nats_user:
            opts["user"] = self.nats_user
            opts["password"] = self.nats_password

        try:
            nc = await nats.connect(**opts)
        except Exception as exc:
            _LOGGER.error(
                "NovaDriver[%s]: NATS connect failed url=%s: %s",
                self.controller,
                self.nats_url,
                exc,
            )
            return  # thread exits; outer connect() can be retried
        _LOGGER.info(
            "NovaDriver[%s]: NATS connected url=%s",
            self.controller,
            self.nats_url,
        )

        async def _cb(msg: Any) -> None:
            try:
                payload = json.loads(msg.data.decode("utf-8"))
            except (ValueError, AttributeError, UnicodeDecodeError):
                return
            with self._lock:
                self._latest = payload
                self._last_update = time.monotonic()

        try:
            await nc.subscribe(self.subject, cb=_cb)
        except Exception as exc:
            _LOGGER.error(
                "NovaDriver[%s]: NATS subscribe to %s failed: %s",
                self.controller,
                self.subject,
                exc,
            )
            with contextlib.suppress(Exception):
                await nc.drain()
            return
        _LOGGER.info(
            "NovaDriver[%s]: subscribed to %s; waiting for state messages "
            "(silent means no client has opened a motion-group session yet)",
            self.controller,
            self.subject,
        )

        assert self._stop_event is not None

        async def _no_data_watchdog() -> None:
            # Warn once if nothing arrives in the first ~5 s so a silent
            # stream is obvious in the logs.
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                return
            with self._lock:
                got = self._last_update > 0.0
            if not got:
                _LOGGER.warning(
                    "NovaDriver[%s]: no NATS messages on %s after 5 s. "
                    "Nova v2 only publishes controller state while a "
                    "motion-group session is active. Check that a Nova "
                    "program or client is currently running against this "
                    "controller.",
                    self.controller,
                    self.subject,
                )

        watchdog = asyncio.ensure_future(_no_data_watchdog())
        try:
            await self._stop_event.wait()
        finally:
            watchdog.cancel()
            with contextlib.suppress(Exception):
                await nc.drain()


register_state_driver("nova", NovaDriver)
