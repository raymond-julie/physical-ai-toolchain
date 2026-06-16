"""Application orchestrator for ur_dual_recorder (recording-only).

Wires the follower arms (read-only state) and cameras into one process:

* opens every configured Orbbec camera,
* reads each follower arm's joints + Robotiq gripper state (no commanding),
* records synchronized LeRobot episodes,
* serves a status/preview dashboard.

Mirroring/teleoperation is intentionally shelved here — nothing is sent to any
robot. Recording is toggled by the web Record button or, optionally, a tool DI0
button on the first follower arm.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .arm_reader import ArmReader
from .cameras import CameraManager
from .config import AppConfig
from .recorder import EpisodeRecorder, RecorderFrame

_LOGGER = logging.getLogger(__name__)


class DualRecorderApp:
    """Top-level coordinator for the recording-only pipeline."""

    def __init__(self, config: AppConfig, enable_di0_trigger: bool = True) -> None:
        self.config = config
        self.enable_di0_trigger = enable_di0_trigger

        # Optional USB foot-pedal listener (set by the entrypoint). Exposed in
        # status() so the dashboard can show per-pedal input activity.
        self.pedal = None

        # Cache for the (slow) recordings-folder size walk so /api/status stays
        # fast. The walk runs in a background thread; status() only ever reads
        # the cached value and never blocks on the filesystem.
        self._recsize_cache = 0
        self._recsize_at = 0.0
        self._recsize_ttl = 30.0
        self._recsize_busy = False
        self._recsize_lock = threading.Lock()

        # Follower arms, ordered by side for a stable schema.
        followers = sorted(
            (a for a in config.arms.values() if a.mode == "follower"),
            key=lambda a: (a.side, a.device_id),
        )
        if not followers:
            # Fall back to the followers referenced by the configured pairs.
            followers = [p.follower for p in config.pairs]

        self.arm_readers: list[ArmReader] = [
            ArmReader(name=a.device_id, ip=a.ip, gripper_cfg=config.gripper)
            for a in followers
        ]
        self.arm_meta = {a.device_id: a for a in followers}
        self.arm_names = [r.name for r in self.arm_readers]

        self.cameras = CameraManager(config.cameras, config.camera)
        self.camera_ids = self.cameras.device_ids()

        self.recorder: EpisodeRecorder | None = None
        if config.recording.get("enabled", True):
            self.recorder = EpisodeRecorder(
                arm_names=self.arm_names,
                camera_ids=self.camera_ids,
                frame_provider=self._build_frame,
                recording_cfg=config.recording,
            )

        self._stop = threading.Event()

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        _LOGGER.info("Opening cameras...")
        self.cameras.open_all()
        _LOGGER.info("Connecting follower arm readers + grippers...")
        for reader in self.arm_readers:
            reader.connect()
            reader.start()
        if self.recorder is not None and self.recorder.open():
            self.recorder.start()

    def shutdown(self) -> None:
        _LOGGER.info("Shutting down...")
        self._stop.set()
        if self.recorder is not None:
            self.recorder.shutdown()
        for reader in self.arm_readers:
            reader.shutdown()
        self.cameras.stop_all()

    # ── recording control ───────────────────────────────────────────────
    def toggle_recording(self) -> None:
        if self.recorder is None:
            _LOGGER.warning("Recorder disabled; toggle ignored.")
            return
        self.recorder.toggle_episode()

    def start_recording(self) -> None:
        if self.recorder is None:
            _LOGGER.warning("Recorder disabled; start ignored.")
            return
        self.recorder.start_episode()

    def stop_recording(self) -> None:
        if self.recorder is None:
            _LOGGER.warning("Recorder disabled; stop ignored.")
            return
        self.recorder.stop_episode()

    def discard_recording(self) -> None:
        """Stop the current episode and delete it without saving."""
        if self.recorder is None:
            _LOGGER.warning("Recorder disabled; discard ignored.")
            return
        self.recorder.discard_episode()

    def set_defer_encoding(self, enabled: bool) -> None:
        """Toggle deferred encoding: hold stopped episodes raw until requested."""
        if self.recorder is None:
            _LOGGER.warning("Recorder disabled; defer toggle ignored.")
            return
        self.recorder.set_defer_encoding(enabled)

    def encode_pending(self) -> int:
        """Flush all held raw episodes into the encoder. Returns count queued."""
        if self.recorder is None:
            _LOGGER.warning("Recorder disabled; encode ignored.")
            return 0
        return self.recorder.encode_pending()

    @property
    def is_recording(self) -> bool:
        return self.recorder is not None and self.recorder.is_recording

    # ── frame assembly for the recorder ─────────────────────────────────
    def _build_frame(self) -> RecorderFrame | None:
        state_parts = []
        grippers_closed = {}
        for reader in self.arm_readers:
            s = reader.sample()
            state_parts.extend(s.joints[:6])
            state_parts.append(s.gripper_position)
            grippers_closed[reader.name] = s.gripper_is_closed

        images = {}
        for cam_id in self.camera_ids:
            frame = self.cameras.get_frame(cam_id)
            if frame is not None:
                images[cam_id] = frame
        return RecorderFrame(
            state=np.asarray(state_parts, dtype=np.float32),
            grippers_closed=grippers_closed,
            images=images,
        )

    # ── status snapshot for the GUI ─────────────────────────────────────
    def _recordings_size(self) -> int:
        """Return the cached recordings-folder size (bytes), never blocking.

        The actual tree walk is expensive (thousands of files, several GB), so it
        runs in a background thread. ``/api/status`` is polled often and must
        never block on the filesystem; a slightly stale size is fine. Returns 0
        until the first background walk completes.
        """
        now = time.monotonic()
        with self._recsize_lock:
            stale = now - self._recsize_at >= self._recsize_ttl
            if stale and not self._recsize_busy:
                self._recsize_busy = True
                threading.Thread(
                    target=self._recsize_worker, name="recsize", daemon=True
                ).start()
            return self._recsize_cache

    def _recsize_worker(self) -> None:
        """Walk the recordings tree once and update the cached size."""
        try:
            root = (
                self.recorder.root
                if self.recorder is not None
                else Path("./recordings_lerobot")
            )
            root = Path(root)
            total = 0
            if root.exists():
                try:
                    for path in root.rglob("*"):
                        try:
                            if path.is_file() and not path.is_symlink():
                                total += path.stat().st_size
                        except OSError:
                            continue
                except OSError:
                    pass
            self._recsize_cache = total
        finally:
            with self._recsize_lock:
                self._recsize_at = time.monotonic()
                self._recsize_busy = False

    def _disk_usage(self) -> dict:
        """Free / used space on the filesystem holding the recordings."""
        path = self.recorder.root if self.recorder is not None else Path(".")
        # Walk up to the nearest existing directory so disk_usage never fails on a
        # recordings root that has not been created yet.
        probe = Path(path)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            total, used, free = shutil.disk_usage(probe)
        except OSError:
            return {"total": 0, "used": 0, "free": 0, "percent": 0.0}
        return {
            "total": total,
            "used": used,
            "free": free,
            "percent": round(used / total * 100.0, 1) if total else 0.0,
            "recordings": self._recordings_size(),
        }

    def status(self) -> dict:
        arms_status = []
        for reader in self.arm_readers:
            s = reader.sample()
            meta = self.arm_meta.get(reader.name)
            arms_status.append(
                {
                    "name": reader.name,
                    "side": meta.side if meta else "",
                    "ip": reader.ip,
                    "arm_connected": s.arm_connected,
                    "gripper_connected": s.gripper_connected,
                    "gripper_position": round(s.gripper_position, 3),
                    "gripper_is_closed": s.gripper_is_closed,
                    "di0": s.di0,
                    "joints": [round(v, 4) for v in s.joints],
                }
            )
        return {
            "recording": self.is_recording,
            "state": self.recorder.state if self.recorder else "idle",
            "encoding": self.recorder.is_encoding if self.recorder else False,
            "defer_encoding": self.recorder.defer_encoding if self.recorder else False,
            "pending_encode": self.recorder.pending_count if self.recorder else 0,
            "episode_count": self.recorder.episode_count if self.recorder else 0,
            "frames": self.recorder.frames_in_episode if self.recorder else 0,
            "min_frames": self.recorder.min_frames if self.recorder else 0,
            "elapsed": round(self.recorder.recording_seconds, 1)
            if self.recorder
            else 0.0,
            "episodes": self.recorder.episodes if self.recorder else [],
            "cameras": self.camera_ids,
            "arms": arms_status,
            "pedal": self.pedal.state() if self.pedal is not None else None,
            "disk": self._disk_usage(),
        }

    # ── main loop ───────────────────────────────────────────────────────
    def run(self) -> None:
        """Poll the optional DI0 trigger and log status until stopped."""
        last_log = 0.0
        trigger = (
            self.arm_readers[0]
            if (self.enable_di0_trigger and self.arm_readers)
            else None
        )
        while not self._stop.is_set():
            if trigger is not None and trigger.di0_edge.is_set():
                trigger.di0_edge.clear()
                _LOGGER.info("[%s] tool DI0 edge -> toggle recording", trigger.name)
                self.toggle_recording()
            now = time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                self._log_status()
            time.sleep(0.02)

    def _log_status(self) -> None:
        parts = []
        for reader in self.arm_readers:
            s = reader.sample()
            parts.append(
                f"{reader.name}[A:{'ok' if s.arm_connected else 'DOWN'} "
                f"G:{'ok' if s.gripper_connected else 'DOWN'} "
                f"grip={s.gripper_position:.2f}"
                f"{'/closed' if s.gripper_is_closed else '/open'}]"
            )
        _LOGGER.info(
            "rec=%s eps=%d | %s",
            self.is_recording,
            self.recorder.episode_count if self.recorder else 0,
            " ".join(parts),
        )

    def request_stop(self) -> None:
        self._stop.set()
