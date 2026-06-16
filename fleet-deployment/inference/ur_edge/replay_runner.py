#!/usr/bin/env python3
"""Replay runner for the UR edge runtime.

Streams a previously-recorded LeRobot episode back through the
``/mirror/joint_states`` and ``/mirror/gripper/position`` topics so the
``destination_writer`` state machine drives the follower robot through the same
motion. Designed to mirror the structure of ``model_runner.py`` so the GUI can
use the same publish callback and policy-active gating.

Workflow at the GUI level::

    runner = ReplayRunner(
        recordings_dir=Path("recordings_lerobot"),
        publish_action=ros_node.publish_model_action,
        logger=ros_node.get_logger(),
    )
    runner.list_sessions()                     # populate dropdown
    runner.list_episodes(session_name)         # populate episode dropdown
    runner.start(session_name, episode_index)  # GUI also flips /policy/active
    runner.stop()                              # GUI also flips /policy/active

Threading model: a single daemon worker thread does the streaming. Stop sets an
event the worker checks each frame; ``stop()`` joins with a short timeout. State
is locked behind ``self._lock``; ``status()`` is safe to call from the Flask
request handler.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

_LOGGER = logging.getLogger(__name__)

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

DEFAULT_FPS = 15.0

# Supported replay command modes. ``"joints"`` publishes the recorded 6-DOF
# joint targets on ``/mirror/joint_states`` (UR firmware does no IK). ``"tcp"``
# publishes the recorded base-frame TCP pose on ``/mirror/tcp_pose`` so
# destination_writer can drive the follower with ``servoL`` and let the UR
# controller's firmware do the IK.
MODE_JOINTS = "joints"
MODE_TCP = "tcp"


def _load_episode_data(parquet_path: Path,
                       episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(joint_actions[N,7], tcp_actions[N,7], gripper[N])`` for one
    episode.

    Reads the dataset parquet file directly via pyarrow to avoid pulling in the
    full ``LeRobotDataset`` import (which can be slow and pulls in torch). Raises
    ``ValueError`` if the episode is missing. ``tcp_actions`` carries
    ``[x, y, z, rx, ry, rz, gripper]`` (URScript convention, axis-angle).
    """
    df = pq.read_table(parquet_path).to_pandas()
    available = sorted(df.episode_index.unique())
    if episode not in available:
        raise ValueError(
            f"Episode {episode} not in dataset; available: {available}")
    ep = df[df.episode_index == episode].sort_values("frame_index")
    actions = np.stack(ep["action"].values).astype(np.float32)
    tcp = np.stack(ep["action.tcp_pose"].values).astype(np.float32)
    grip = ep["action.gripper.position"].values.astype(np.float32)
    return actions, tcp, grip


def _summarise_session(session_dir: Path) -> dict | None:
    """Best-effort metadata for one session directory.

    Returns ``None`` for directories that don't look like a finalised LeRobot
    dataset (e.g. one that crashed before finalize() and has a corrupt parquet
    footer). The episode list is read from the data parquet (single source of
    truth), not from meta/episodes (which can be a subset if save_episode
    crashed).
    """
    data_files = sorted((session_dir / "data").rglob("file-*.parquet"))
    if not data_files:
        return None
    try:
        df = pq.read_table(data_files[0]).to_pandas()
    except Exception:
        return None
    episodes = []
    for ep in sorted(df.episode_index.unique()):
        s = df[df.episode_index == ep]
        ts = s["timestamp"].values
        episodes.append({
            "episode": int(ep),
            "frames": int(len(s)),
            "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        })
    return {
        "name": session_dir.name,
        "episodes": episodes,
        "parquet_path": str(data_files[0]),
    }


def list_sessions(recordings_dir: Path) -> list[dict]:
    """Enumerate every replayable session under ``recordings_dir``.

    Sessions are sorted newest first by directory name (which embeds a sortable
    timestamp suffix). Sessions whose parquet is unreadable are silently skipped
    — they cannot be replayed anyway.
    """
    out: list[dict] = []
    if not recordings_dir.exists():
        return out
    for d in sorted(recordings_dir.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("session_"):
            continue
        info = _summarise_session(d)
        if info is not None:
            out.append(info)
    return out


class ReplayRunner:
    STATE_STOPPED = "stopped"
    STATE_LOADING = "loading"
    STATE_RUNNING = "running"

    def __init__(self,
                 recordings_dir: Path,
                 publish_action: Callable[[list[float], float], None],
                 logger: Any = None,
                 fps: float = DEFAULT_FPS,
                 publish_tcp: Callable[[list[float], float], None] | None = None) -> None:
        """:param publish_action: same callback as ``ModelRunner`` —
        ``(joint_targets[6], gripper_pos[0..1])``. Invoked from the worker
        thread once per frame at ``fps`` Hz when mode="joints".
        :param publish_tcp: callback ``(tcp_pose[6], gripper_pos[0..1])`` used
        when mode="tcp". Optional — if ``None``, only joint replay is available.
        """
        self.recordings_dir = Path(recordings_dir)
        self.publish_action = publish_action
        self.publish_tcp = publish_tcp
        self.logger = logger
        self.fps = fps

        self._lock = threading.Lock()
        self._state = self.STATE_STOPPED
        self._message = ""
        self._session: str | None = None
        self._episode: int | None = None
        self._mode: str = MODE_JOINTS

        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        # Stats updated from the worker.
        self._frame: int = 0
        self._total_frames: int = 0
        self._fps_actual: float = 0.0
        # Notified when the worker thread exits so the GUI can flip
        # /policy/active back to False without polling.
        self._on_finished: Callable[[], None] | None = None

    # ── public API ─────────────────────────────────────────────────

    def set_on_finished(self, fn: Callable[[], None] | None) -> None:
        """Register a callback invoked when the worker thread exits (whether by
        completing the episode or by stop())."""
        self._on_finished = fn

    def list_sessions(self) -> list[dict]:
        return list_sessions(self.recordings_dir)

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "message": self._message,
                "session": self._session,
                "episode": self._episode,
                "mode": self._mode,
                "frame": self._frame,
                "total_frames": self._total_frames,
                "fps_actual": round(self._fps_actual, 2),
                "fps_target": self.fps,
            }

    def start(self, session: str, episode: int,
              mode: str = MODE_JOINTS) -> dict:
        """Begin replaying ``episode`` of ``session``.

        :param mode: one of ``"joints"`` (publish joint targets, no IK) or
            ``"tcp"`` (publish Cartesian pose, UR controller does the IK via
            ``servoL``). Refuses to start in ``"tcp"`` mode if no ``publish_tcp``
            callback was provided.

        Refuses if a replay is already loading or running. Returns a dict
        ``{ok, msg}`` for direct use as a Flask JSON response.
        """
        mode = str(mode or MODE_JOINTS).lower()
        if mode not in (MODE_JOINTS, MODE_TCP):
            return {"ok": False,
                    "msg": f'mode must be "joints" or "tcp", got {mode!r}'}
        if mode == MODE_TCP and self.publish_tcp is None:
            return {"ok": False,
                    "msg": "TCP replay not wired: no publish_tcp callback"}

        with self._lock:
            if self._state in (self.STATE_LOADING, self.STATE_RUNNING):
                return {"ok": False,
                        "msg": f"Replay already {self._state}"}

        session_dir = self.recordings_dir / session
        if not session_dir.is_dir():
            return {"ok": False, "msg": f"Session not found: {session}"}
        data_files = sorted((session_dir / "data").rglob("file-*.parquet"))
        if not data_files:
            return {"ok": False,
                    "msg": f"No parquet under {session}/data"}
        try:
            actions, tcp, grip = _load_episode_data(
                data_files[0], int(episode))
        except Exception as exc:
            return {"ok": False, "msg": f"Load failed: {exc}"}

        with self._lock:
            self._state = self.STATE_LOADING
            self._session = session
            self._episode = int(episode)
            self._mode = mode
            self._frame = 0
            self._total_frames = len(actions)
            self._fps_actual = 0.0
            self._message = (f"Loaded {len(actions)} frames "
                             f"({len(actions) / self.fps:.1f}s, {mode})")

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, args=(actions, tcp, grip),
            name=f"ReplayRunner-{session}-ep{episode}-{mode}",
            daemon=True)
        self._thread.start()
        return {"ok": True,
                "msg": f"Replaying {session} episode {episode} ({mode})"}

    def stop(self) -> dict:
        with self._lock:
            if self._state == self.STATE_STOPPED:
                return {"ok": True, "msg": "Already stopped"}
        self._stop_evt.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        with self._lock:
            self._state = self.STATE_STOPPED
            self._message = "Stopped"
        return {"ok": True, "msg": "Replay stopped"}

    # ── worker ─────────────────────────────────────────────────────

    def _log(self, level: str, msg: str) -> None:
        if self.logger is None:
            _LOGGER.log(_LOG_LEVELS.get(level.lower(), logging.INFO),
                        "[replay] %s", msg)
            return
        try:
            from rclpy.logging import LoggingSeverity
            sev_map = {
                "debug": LoggingSeverity.DEBUG,
                "info": LoggingSeverity.INFO,
                "warn": LoggingSeverity.WARN,
                "warning": LoggingSeverity.WARN,
                "error": LoggingSeverity.ERROR,
            }
            sev = sev_map.get(level.lower(), LoggingSeverity.INFO)
            self.logger.log(f"[replay] {msg}", sev)
        except Exception:
            getattr(self.logger, level, self.logger.info)(f"[replay] {msg}")

    def _run(self, actions: np.ndarray, tcp: np.ndarray, grip: np.ndarray) -> None:
        """Worker thread body — streams frames at ``fps``.

        The first ~1 s holds the start *joint* pose regardless of mode so
        destination_writer's catch-up phase can interpolate the follower from
        its current pose to the episode start at ``alignment_speed`` before any
        trajectory streaming begins. For TCP mode we then switch to
        ``/mirror/tcp_pose`` and the UR controller's firmware takes over IK.
        """
        n = len(actions)
        dt = 1.0 / max(self.fps, 1.0)
        tcp_mode = (self._mode == MODE_TCP)

        with self._lock:
            self._state = self.STATE_RUNNING
            self._message = (
                f"Streaming {n} frames @ {self.fps:.1f} Hz ({self._mode})")
        self._log("info",
                  f"start session={self._session} episode={self._episode} "
                  f"frames={n} fps={self.fps} mode={self._mode}")

        try:
            # Start-pose hold (catch-up window). Always publish joints for this
            # phase — the follower may be anywhere (home or mid-trajectory from
            # a previous run) and destination_writer knows how to interpolate
            # joints toward the target. Once we begin streaming, TCP mode takes
            # over.
            hold_frames = int(round(1.0 * self.fps))
            for _ in range(hold_frames):
                if self._stop_evt.is_set():
                    break
                self.publish_action(
                    list(actions[0, :6]), float(grip[0]))
                time.sleep(dt)

            t0 = time.monotonic()
            for i in range(n):
                if self._stop_evt.is_set():
                    self._log("info", f"stop requested at frame {i}/{n}")
                    break
                target = t0 + i * dt
                now = time.monotonic()
                if target > now:
                    time.sleep(target - now)
                if tcp_mode:
                    self.publish_tcp(
                        list(tcp[i, :6]), float(grip[i]))
                else:
                    self.publish_action(
                        list(actions[i, :6]), float(grip[i]))
                with self._lock:
                    self._frame = i + 1
                    elapsed = max(time.monotonic() - t0, 1e-3)
                    self._fps_actual = (i + 1) / elapsed

            # Final pose hold so the follower fully settles before the GUI flips
            # /policy/active to False (which triggers RETURNING).
            if not self._stop_evt.is_set():
                for _ in range(int(round(0.5 * self.fps))):
                    if tcp_mode:
                        self.publish_tcp(
                            list(tcp[-1, :6]), float(grip[-1]))
                    else:
                        self.publish_action(
                            list(actions[-1, :6]), float(grip[-1]))
                    time.sleep(dt)
            self._log("info", "replay finished")
        except Exception as exc:
            self._log("error", f"worker crashed: {exc}")
            with self._lock:
                self._message = f"Crashed: {exc}"
        finally:
            with self._lock:
                self._state = self.STATE_STOPPED
                if self._stop_evt.is_set():
                    self._message = "Stopped"
                else:
                    self._message = "Finished"
            cb = self._on_finished
            if cb is not None:
                try:
                    cb()
                except Exception as exc:
                    self._log("warn", f"on_finished callback raised: {exc}")
