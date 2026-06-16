"""USB foot-pedal trigger for toggling recording.

The iKKEGOL USB foot switch (and most programmable foot pedals) enumerates as a
standard USB HID **keyboard**: pressing the pedal emits a key-down event for a
configured key, releasing it emits key-up. This module listens to that input
device directly via the Linux ``evdev`` interface so it works headless (no X11,
no focused window required) on the Jetson.

Each pedal press fires a callback, which the app wires to the recorder's
start/stop/discard controls — so one tap starts recording and the next stops it.

The dependency (``python-evdev``) and a connected pedal are both optional: if
either is missing the listener degrades gracefully and logs why, leaving the web
Record button and DI0 trigger as fallbacks.
"""

from __future__ import annotations

import contextlib
import logging
import select
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evdev import InputDevice

_LOGGER = logging.getLogger(__name__)

try:
    from evdev import InputDevice, categorize, ecodes, list_devices

    EVDEV_AVAILABLE = True
except ImportError:  # pragma: no cover - evdev only present on the lab host
    EVDEV_AVAILABLE = False


# Substrings (case-insensitive) used to auto-detect a foot pedal by device name
# when no explicit device path/name is configured.
_PEDAL_NAME_HINTS = ("pedal", "foot", "footswitch", "switch")


def list_input_devices() -> list[dict]:
    """Return metadata for every readable input device.

    Useful for identifying which ``/dev/input/eventN`` is the foot pedal: run
    ``python -m ur_dual_recorder --list-pedals`` and look for the entry that
    appears/disappears when you unplug the pedal.
    """
    if not EVDEV_AVAILABLE:
        return []
    devices = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except (OSError, PermissionError):
            continue
        has_keys = ecodes.EV_KEY in dev.capabilities()
        devices.append(
            {
                "path": dev.path,
                "name": dev.name,
                "phys": dev.phys,
                "has_keys": has_keys,
            }
        )
        dev.close()
    return devices


class FootPedalListener:
    """Listen on a USB foot pedal and trigger recording actions per key.

    A multi-pedal foot switch emits a different key for each pedal. Map each key
    to its own callback via ``key_actions`` (e.g. ``KEY_A`` -> start, ``KEY_B``
    -> stop, ``KEY_C`` -> discard). For a single-pedal toggle, pass ``on_press``
    instead (called on every key-down, or only the ``key`` one).

    Args:
        on_press: Fallback callback for every (or the configured ``key``)
            key-down when no ``key_actions`` entry matches.
        key_actions: Map of key name -> callback. Takes priority over
            ``on_press`` for matching keys.
        key_labels: Optional map of key name -> human label (shown in the UI).
        device_path: Explicit ``/dev/input/eventN`` to use. Takes priority.
        device_name: Case-insensitive substring matched against device names when
            ``device_path`` is not given. Empty -> auto-detect by hints.
        key: Optional key name (e.g. ``"KEY_B"``) the ``on_press`` fallback reacts
            to. Empty -> react to any key-down.
        grab: If True, grab the device exclusively so the pedal's keystroke does
            not leak to the rest of the system.
    """

    def __init__(
        self,
        on_press: Callable[[], None] | None = None,
        *,
        key_actions: dict[str, Callable[[], None]] | None = None,
        key_labels: dict[str, str] | None = None,
        device_path: str | None = None,
        device_name: str | None = None,
        key: str | None = None,
        grab: bool = True,
    ) -> None:
        self.on_press = on_press
        self.key_actions = {
            k.strip().upper(): v for k, v in (key_actions or {}).items()
        }
        self.key_labels = {
            k.strip().upper(): v for k, v in (key_labels or {}).items()
        }
        self.device_path = device_path or None
        self.device_name = (device_name or "").strip()
        self.key = (key or "").strip().upper() or None
        self.grab = grab

        self._dev: InputDevice | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Per-pedal (per-key) activity tracking for the UI.
        self._lock = threading.Lock()
        self._key_counts: dict[str, int] = {}
        self._key_last_ts: dict[str, float] = {}
        self._total_presses = 0
        self._last_key: str | None = None

    @property
    def connected(self) -> bool:
        """True once a pedal device has been opened and the reader started."""
        return self._dev is not None and self._thread is not None

    def state(self, active_window: float = 0.6) -> dict:
        """Snapshot of pedal activity for the dashboard.

        Each distinct key the foot switch emits is reported as one "pedal" with a
        press count and an ``active`` flag that lights up briefly after each press
        (within ``active_window`` seconds).
        """
        now = time.monotonic()
        with self._lock:
            # Show configured pedals even before they are first pressed.
            known = set(self._key_counts) | set(self.key_labels) | set(self.key_actions)
            keys = [
                {
                    "key": key,
                    "label": self.key_labels.get(key, ""),
                    "count": self._key_counts.get(key, 0),
                    "active": (now - self._key_last_ts.get(key, 0.0)) <= active_window,
                    "age": round(now - self._key_last_ts.get(key, now), 2),
                }
                for key in sorted(known)
            ]
            return {
                "connected": self.connected,
                "device": self._dev.name if self._dev is not None else "",
                "path": self._dev.path if self._dev is not None else "",
                "total_presses": self._total_presses,
                "last_key": self._last_key,
                "keys": keys,
            }

    def _wanted_key_codes(self) -> list[int]:
        """evdev codes for the keys we actually react to (for ranking)."""
        names = list(self.key_actions) + ([self.key] if self.key else [])
        codes = []
        for name in names:
            code = ecodes.ecodes.get(name)
            if isinstance(code, int):
                codes.append(code)
        return codes

    def _pick_key_device(self, devices: list[InputDevice]) -> InputDevice | None:
        """Choose the interface most likely to emit our pedal keys.

        Multi-interface USB foot switches expose keyboard/mouse/raw nodes that
        share a name. Rank by: (1) advertises the configured action keys,
        (2) name contains "keyboard", (3) raw count of keyboard-range keys.
        """
        if not devices:
            return None
        if len(devices) == 1:
            return devices[0]

        wanted = set(self._wanted_key_codes())

        def score(dev: InputDevice) -> tuple[int, int, int]:
            keys = set(dev.capabilities().get(ecodes.EV_KEY, []))
            has_wanted = 1 if (wanted and wanted & keys) else 0
            is_kbd = 1 if "keyboard" in dev.name.lower() else 0
            letter_keys = sum(1 for c in keys if ecodes.KEY_A <= c <= ecodes.KEY_Z)
            return (has_wanted, is_kbd, letter_keys)

        return max(devices, key=score)

    def _resolve_device(self) -> InputDevice | None:
        if self.device_path:
            try:
                return InputDevice(self.device_path)
            except (OSError, PermissionError) as exc:
                _LOGGER.error("Foot pedal: cannot open %s (%s)", self.device_path, exc)
                return None

        candidates = []
        for path in list_devices():
            try:
                dev = InputDevice(path)
            except (OSError, PermissionError):
                continue
            if ecodes.EV_KEY not in dev.capabilities():
                dev.close()
                continue
            candidates.append(dev)

        # Narrow to name matches (explicit device_name, then generic hints).
        matches = candidates
        if self.device_name:
            needle = self.device_name.lower()
            named = [d for d in candidates if needle in d.name.lower()]
            if named:
                matches = named
        if matches is candidates:
            hinted = [
                d
                for d in candidates
                if any(h in d.name.lower() for h in _PEDAL_NAME_HINTS)
            ]
            if hinted:
                matches = hinted

        # A multi-interface foot switch exposes several event nodes (keyboard,
        # mouse, raw). Prefer the one that actually emits the keys we react to, so
        # auto-detect doesn't latch onto the mouse interface.
        chosen = self._pick_key_device(matches) if matches else None

        for dev in candidates:
            if dev is not chosen:
                dev.close()

        if chosen is None:
            _LOGGER.warning(
                "Foot pedal: no matching input device found. "
                "Run `python -m ur_dual_recorder --list-pedals` to identify it, "
                "then set pedal.device_name or pedal.device_path in app.yaml."
            )
        return chosen

    def start(self) -> bool:
        if not EVDEV_AVAILABLE:
            _LOGGER.warning(
                "Foot pedal: python-evdev not installed; pedal disabled "
                "(pip install evdev)."
            )
            return False

        self._dev = self._resolve_device()
        if self._dev is None:
            return False

        if self.grab:
            try:
                self._dev.grab()
            except OSError as exc:
                _LOGGER.warning(
                    "Foot pedal: could not grab %s exclusively (%s); "
                    "the keystroke may also reach the system.",
                    self._dev.path,
                    exc,
                )

        self._thread = threading.Thread(
            target=self._run, name="foot-pedal", daemon=True
        )
        self._thread.start()
        _LOGGER.info(
            "Foot pedal ready on %s (%s); press to toggle recording%s.",
            self._dev.path,
            self._dev.name,
            f" [key {self.key}]" if self.key else "",
        )
        return True

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._dev is not None:
            if self.grab:
                with contextlib.suppress(OSError):
                    self._dev.ungrab()
            self._dev.close()
            self._dev = None

    def _run(self) -> None:
        dev = self._dev
        assert dev is not None
        while not self._stop.is_set():
            try:
                # Wake periodically so shutdown() is responsive.
                ready, _, _ = select.select([dev.fd], [], [], 0.5)
                if not ready:
                    continue
                for event in dev.read():
                    if event.type != ecodes.EV_KEY:
                        continue
                    # value: 1 = key-down, 0 = key-up, 2 = autorepeat.
                    if event.value != 1:
                        continue
                    self._on_key(event)
            except OSError as exc:
                _LOGGER.error("Foot pedal: device read error (%s); stopping.", exc)
                break

    def _on_key(self, event: Any) -> None:
        try:
            name = categorize(event).keycode
        except Exception:
            name = event.code
        if isinstance(name, (list, tuple)):
            name = name[0]
        name = str(name).upper()
        with self._lock:
            self._key_counts[name] = self._key_counts.get(name, 0) + 1
            self._key_last_ts[name] = time.monotonic()
            self._total_presses += 1
            self._last_key = name

        # Resolve the action: a per-key mapping wins; otherwise fall back to
        # on_press for the configured key (or any key when none is set).
        action = self.key_actions.get(name)
        if action is None and self.on_press is not None and (
            self.key is None or name == self.key
        ):
            action = self.on_press

        label = self.key_labels.get(name, "")
        if action is None:
            _LOGGER.info(
                "Foot pedal press (%s) — no action bound (tracked only)", name
            )
            return
        _LOGGER.info("Foot pedal press (%s)%s", name, f" -> {label}" if label else "")
        try:
            action()
        except Exception:
            _LOGGER.exception("Foot pedal: action for %s failed", name)
