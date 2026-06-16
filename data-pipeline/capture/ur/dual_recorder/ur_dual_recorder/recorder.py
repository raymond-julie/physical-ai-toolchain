"""Multi-arm LeRobotDataset episode recorder (recording-only).

Records synchronized episodes of read-only follower-arm state plus camera
frames. No robot is commanded, so there is no ``action`` stream — only
observations. The frame schema is built from the live configuration:

* ``observation.state``                   — per arm: 6 joints + gripper position
  (closed fraction 0..1).
* ``observation.<arm>.gripper_is_closed`` — bool, per arm.
* ``observation.images.<camera_id>``      — one RGB video stream per camera.

The recorder owns a sampling thread. While an episode is active it pulls the
latest frame from a provider callback at ``fps`` and appends it. Episode
boundaries are driven externally (web Record button or a follower tool DI0) via
:meth:`start_episode` / :meth:`stop_episode`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# HuggingFace Hub offline — local repo ids must never trigger a network lookup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_LOGGER = logging.getLogger(__name__)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    try:  # older lerobot layout
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        LEROBOT_AVAILABLE = True
    except ImportError:  # pragma: no cover
        LEROBOT_AVAILABLE = False
        LeRobotDataset = None

# Dataset on-disk format version of the bundled lerobot ("v2.1" or "v3.0").
# Selected at launch by activating the matching venv (.venv vs .venv-v3); the
# recorder adapts its create()/codec path to whichever build is importable.
try:
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION as _LEROBOT_CODEBASE
except Exception:
    try:
        from lerobot.common.datasets.lerobot_dataset import (
            CODEBASE_VERSION as _LEROBOT_CODEBASE,
        )
    except Exception:
        _LEROBOT_CODEBASE = "v2.1"


def _normalize_format(value: object) -> str:
    """Canonicalize a LeRobot format label to ``"v2.1"`` or ``"v3.0"``."""
    s = str(value).strip().lower().lstrip("v")
    if s.startswith("3"):
        return "v3.0"
    return "v2.1"


class RecorderFrame:
    """A single synchronized observation sample."""

    __slots__ = ("grippers_closed", "images", "state")

    def __init__(
        self,
        state: np.ndarray,
        grippers_closed: dict[str, bool],
        images: dict[str, np.ndarray],
    ) -> None:
        self.state = state
        self.grippers_closed = grippers_closed
        self.images = images


# Provider returns the latest RecorderFrame (or None if not ready).
FrameProvider = Callable[[], "RecorderFrame | None"]


class EpisodeRecorder:
    """Streams synchronized observation frames into LeRobotDataset episodes."""

    def __init__(
        self,
        arm_names: list[str],
        camera_ids: list[str],
        frame_provider: FrameProvider,
        recording_cfg: dict,
    ) -> None:
        self.arm_names = arm_names
        self.camera_ids = camera_ids
        self.frame_provider = frame_provider
        self.cfg = recording_cfg

        self.fps = int(recording_cfg.get("fps", 30))
        self.repo_id = recording_cfg.get("repo_id", "local/ur_dual")
        self.root = Path(recording_cfg.get("root", "./recordings_lerobot")).expanduser()
        self.task = recording_cfg.get("task", "dual_arm_recording")
        self.use_videos = bool(recording_cfg.get("use_videos", True))
        self.min_frames = int(recording_cfg.get("min_episode_frames", 5))
        self.image_h = int(recording_cfg.get("image_height", 480))
        self.image_w = int(recording_cfg.get("image_width", 848))
        # Video codec for the per-episode mp4s. lerobot v2.1 defaults to
        # "libsvtav1" (AV1), but GR00T / most downstream decoders expect H.264, so
        # default to "h264" here. Accepts h264 | hevc | libsvtav1.
        self.video_codec = str(recording_cfg.get("video_codec", "h264")).lower()
        # LeRobot dataset on-disk format. The launcher activates the venv that
        # provides the matching lerobot build, so the *running* format is the
        # bundled CODEBASE_VERSION; the configured value is informational and used
        # to warn on a mismatch.
        self.lerobot_format = _normalize_format(
            recording_cfg.get("lerobot_format", _LEROBOT_CODEBASE)
        )
        self.codebase_version = _normalize_format(_LEROBOT_CODEBASE)

        # State dims: per arm 6 joints + 1 gripper position.
        self.dim = len(arm_names) * 7

        self._dataset = None
        self._lock = threading.Lock()
        # add_frame calling convention: v2.1 passes task as a separate arg;
        # resolved against the live lerobot build in open(). Default to the modern
        # (separate-arg) form.
        self._task_as_arg = True
        # Serializes every mutation of the LeRobot dataset buffer. Only the
        # encoder worker thread ever touches the dataset (replay + save_episode);
        # the sampling thread never does, so capture is never blocked by a save.
        self._ds_lock = threading.RLock()
        self._recording = False
        self._frames_in_episode = 0
        # Live-capture buffer: the sampling thread appends fully-formed, RAM-owned
        # frame dicts here (decoupled from the LeRobot dataset). On stop, the list
        # is handed to the encoder worker which replays it into the dataset and
        # saves it OFF the capture path -- so the NEXT episode can start recording
        # immediately while the previous one still encodes. lerobot v2.1
        # save_episode() persists self.episode_buffer in place and cannot
        # double-buffer, which is exactly why capture is kept separate.
        self._live_records: list[dict] = []
        # Deferred-encoding mode: when enabled, a stopped episode is NOT sent to
        # the encoder immediately. Its frames are held in RAM in "raw" state (see
        # _pending_episodes) until the operator presses Encode, which flushes them
        # all into the encode queue at once. Lets a session of back-to-back takes
        # record without competing with ffmpeg for CPU.
        self._defer_encoding = bool(recording_cfg.get("defer_encoding", False))
        self._pending_episodes: list[dict] = []
        # RAM safety valve: cap the total frames held in memory (live + queued for
        # save) by an approximate byte budget so a long episode or a deep encode
        # backlog can't OOM the box. Bytes per frame are computed in open() once
        # the image size + camera count are known.
        self._frame_bytes = 1
        self._max_buffered_frames = 1_000_000
        self._buffered_frames = 0
        self._last_buf_warn = 0.0
        # Episode timing: monotonic start stamp while recording, and the frozen
        # duration of the most recently finished episode.
        self._episode_start_t = 0.0
        self._episode_elapsed = 0.0
        self.episode_count = 0
        # History of finished episodes (most recent first) for the GUI list.
        self._episodes: list[dict] = []
        # Monotonic id so an in-progress (encoding) history entry can be updated in
        # place to saved/error when the background save completes.
        self._history_seq = 0
        # Adaptive encode-time estimate (seconds per frame) used to drive the GUI
        # encoding progress bar. Refined via EMA after every save. Seeded with a
        # realistic per-frame cost so the first bar paces sensibly.
        self._enc_spf = 0.09

        self._thread: threading.Thread | None = None
        # Background save/encode pipeline. A single worker drains save jobs in
        # FIFO order. lerobot v2.1 persists one complete episode parquet + mp4 per
        # save and asserts the on-disk file counts match num_episodes, so saves
        # MUST run one at a time and strictly in order. The web stop request
        # returns immediately (the worker does the heavy ffmpeg encode
        # asynchronously and reports progress), but recording the NEXT episode only
        # resumes once the in-flight save has reset the live buffer.
        self._encode_queue: queue.Queue[dict | None] = queue.Queue()
        self._encoder_worker: threading.Thread | None = None
        self._encode_pending = 0  # queued + in-flight save jobs (GUI)
        self._encode_lock = threading.Lock()
        self._stop = threading.Event()

    # ── schema ──────────────────────────────────────────────────────────
    def _state_names(self) -> list[str]:
        names = []
        for arm in self.arm_names:
            names += [f"{arm}_j{i}" for i in range(6)]
            names += [f"{arm}_gripper"]
        return names

    def _features(self) -> dict:
        feats = {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.dim,),
                "names": self._state_names(),
            },
        }
        for arm in self.arm_names:
            feats[f"observation.{arm}.gripper_is_closed"] = {
                "dtype": "bool",
                "shape": (1,),
                "names": ["is_closed"],
            }
        for cam_id in self.camera_ids:
            feats[f"observation.images.{cam_id}"] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": (self.image_h, self.image_w, 3),
                "names": ["height", "width", "channels"],
            }
        return feats

    # ── dataset lifecycle ───────────────────────────────────────────────
    def open(self) -> bool:
        if not LEROBOT_AVAILABLE:
            _LOGGER.warning("lerobot not installed; recording disabled.")
            return False
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_root = self.root / f"session_{ts}"
        session_root.parent.mkdir(parents=True, exist_ok=True)
        if session_root.exists():
            shutil.rmtree(session_root)
        _LOGGER.info("Creating LeRobotDataset @ %s", session_root)
        # Give lerobot async image-writer threads so add_frame does NOT block on
        # synchronous PNG writes for every camera (which otherwise caps the
        # sampling loop far below the configured fps). ~4 threads per camera
        # comfortably absorbs the per-frame disk I/O at 30 fps.
        n_writer_threads = max(4, 4 * len(self.camera_ids))
        # At save time lerobot encodes the camera videos in parallel (one process
        # per camera). Spread the cores across those encoders, but RESERVE a few
        # cores so the web server, MJPEG previews and the sampling loop stay
        # responsive while a (possibly concurrent) episode encodes — otherwise the
        # whole UI feels laggy under full CPU load.
        n_cams = max(1, len(self.camera_ids))
        n_cpu = os.cpu_count() or 1
        is_v3 = self.codebase_version == "v3.0"
        if self.lerobot_format != self.codebase_version:
            _LOGGER.warning(
                "Configured lerobot_format %s but the active lerobot build is %s; "
                "using %s. Restart with the matching venv to change formats.",
                self.lerobot_format,
                self.codebase_version,
                self.codebase_version,
            )
        # lerobot v2.1 (CODEBASE_VERSION "v2.1", the GR00T-compatible format)
        # writes one COMPLETE episode_NNNNNN.parquet per episode and encodes each
        # episode's videos with a single, internally multithreaded ffmpeg call at
        # save time. It exposes image-writer parallelism for the async PNG staging
        # only -- there are no encoder-thread / metadata-buffer knobs as in the
        # later v3.0 streaming writer.
        create_kwargs = {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "root": session_root,
            "features": self._features(),
            "use_videos": self.use_videos,
            "image_writer_threads": n_writer_threads,
        }
        if is_v3:
            # v3.0's streaming writer takes the codec directly (no monkeypatch),
            # and we encode one episode at a time to keep saves serialized.
            create_kwargs["vcodec"] = self.video_codec
            create_kwargs["batch_encoding_size"] = 1
        # Tolerate API drift across lerobot versions: drop any optional kwarg a
        # given build rejects and retry until create() accepts the rest.
        optional = [
            "image_writer_threads",
            "image_writer_processes",
            "vcodec",
            "batch_encoding_size",
        ]
        while True:
            try:
                self._dataset = LeRobotDataset.create(**create_kwargs)
                break
            except TypeError as exc:
                msg = str(exc)
                dropped = next(
                    (k for k in optional if k in create_kwargs and k in msg), None
                )
                if dropped is None:
                    raise
                create_kwargs.pop(dropped, None)
                _LOGGER.warning(
                    "LeRobotDataset.create rejected %r; retrying without it.", dropped
                )
        # Detect the add_frame calling convention once: v2.1 takes the task as a
        # separate argument (add_frame(frame, task)); older builds expected it as a
        # key inside the frame dict.
        try:
            import inspect

            self._task_as_arg = (
                "task" in inspect.signature(self._dataset.add_frame).parameters
            )
        except (TypeError, ValueError):
            self._task_as_arg = True
        # Size the in-RAM live-capture budget now that the image geometry is known.
        # Each buffered frame holds one uint8 HxWx3 image per camera plus a small
        # float32 state vector; cap the *total* buffered frames (live + awaiting
        # encode) to roughly ``max_buffer_gb`` so a long take or a deep encode
        # backlog degrades gracefully instead of exhausting memory.
        self._frame_bytes = n_cams * self.image_h * self.image_w * 3 + self.dim * 4
        max_gb = float(self.cfg.get("max_buffer_gb", 6.0))
        self._max_buffered_frames = max(
            self.min_frames + 1,
            int(max_gb * (1024**3) / max(1, self._frame_bytes)),
        )
        _LOGGER.info(
            "Recording %s dataset: %d camera(s), %d image-writer threads (%d CPUs). "
            "Live-capture RAM budget %.1f GiB (~%d frames).",
            self.codebase_version,
            n_cams,
            n_writer_threads,
            n_cpu,
            max_gb,
            self._max_buffered_frames,
        )
        # Force the requested video codec (H.264 by default) for episode mp4s, and
        # emit a GR00T-style modality.json describing the dataset layout. v3.0
        # takes the codec via create(vcodec=...) above, so the v2.1 monkeypatch is
        # only needed for the v2.1 build.
        if not is_v3:
            self._install_video_codec()
        else:
            self._validate_codec_available()
        self._write_modality_json(session_root)
        return True

    # ── GR00T compatibility: H.264 encoding + modality.json ─────────────
    def _validate_codec_available(self) -> str:
        """Clamp ``self.video_codec`` to a codec the bundled pyav can encode.

        Returns the (possibly adjusted) codec. Unknown codecs and encoders not
        compiled into the bundled ffmpeg/pyav fall back to ``libsvtav1`` (AV1),
        which lerobot always ships.
        """
        codec = self.video_codec
        if codec not in ("h264", "hevc", "libsvtav1"):
            _LOGGER.warning("Unknown video_codec %r; using libsvtav1.", codec)
            self.video_codec = codec = "libsvtav1"
        # Verify the encoder is actually available in the bundled ffmpeg/pyav.
        try:
            import av

            av.codec.Codec(codec, "w")
        except Exception:
            _LOGGER.warning(
                "Video encoder %r unavailable in pyav; falling back to libsvtav1.",
                codec,
            )
            self.video_codec = codec = "libsvtav1"
        return codec

    def _install_video_codec(self) -> None:
        """Force lerobot's episode encoder to use ``self.video_codec``.

        lerobot v2.1's ``encode_episode_videos`` calls ``encode_video_frames``
        with no ``vcodec`` argument, so it always defaults to ``libsvtav1`` (AV1).
        GR00T and most decoders expect H.264, so we wrap the module-level
        ``encode_video_frames`` reference to inject the codec. Idempotent and falls
        back to the default codec if the requested encoder is unavailable.
        """
        codec = self._validate_codec_available()
        import lerobot.datasets.lerobot_dataset as lds

        original = getattr(
            lds.encode_video_frames, "_codec_original", lds.encode_video_frames
        )
        if codec == "libsvtav1":
            # Restore the stock encoder (AV1 is lerobot's own default).
            lds.encode_video_frames = original
            return

        def patched(
            imgs_dir: Any, video_path: Any, fps: Any, *args: Any, **kwargs: Any
        ) -> Any:
            kwargs.setdefault("vcodec", codec)
            return original(imgs_dir, video_path, fps, *args, **kwargs)

        patched._codec_original = original
        lds.encode_video_frames = patched
        _LOGGER.info("Episode videos will be encoded as %s.", codec)

    def _write_modality_json(self, session_root: Path) -> None:
        """Write ``meta/modality.json`` describing state + video modalities.

        GR00T-compatible LeRobot datasets carry a ``modality.json`` that maps the
        flat ``observation.state`` vector into named sub-fields (by index range)
        and lists each video stream's source column. This recorder has no action
        stream, so only ``state``, ``video`` and ``annotation`` are emitted.
        """
        state = {}
        for i, arm in enumerate(self.arm_names):
            base = i * 7
            # 6 joints, then the gripper position, per arm.
            state[arm] = {"start": base, "end": base + 6}
            state[f"{arm}_gripper"] = {"start": base + 6, "end": base + 7}
        video = {
            cam_id: {"original_key": f"observation.images.{cam_id}"}
            for cam_id in self.camera_ids
        }
        modality = {
            "state": state,
            "video": video,
            "annotation": {
                "human.task_description": {"original_key": "task_index"},
            },
        }
        try:
            meta_dir = session_root / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            import json

            with (meta_dir / "modality.json").open("w", encoding="utf-8") as fh:
                json.dump(modality, fh, indent=2)
            _LOGGER.info("Wrote GR00T modality.json (%s)", meta_dir / "modality.json")
        except Exception as exc:
            _LOGGER.error("Failed to write modality.json: %s", exc)

    def start(self) -> None:
        """Start the sampling thread (idle until an episode is active)."""
        self._stop.clear()
        # Persistent worker that saves/encodes finished episodes in order.
        self._encoder_worker = threading.Thread(
            target=self._encoder_loop, name="recorder-encode", daemon=True
        )
        self._encoder_worker.start()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._recording:
            self.stop_episode()
        # Flush any held 'raw' episodes into the encoder so a deferred-encoding
        # session isn't lost on exit; the drain below waits for them to finish.
        self.encode_pending()
        # Stop sampling first so no new frames are produced.
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Drain any queued/in-flight episode encodes, then retire the worker so we
        # never drop the last recording(s) on exit.
        if self._encoder_worker is not None:
            self._encode_queue.put(None)
            self._encoder_worker.join(timeout=120.0)
            self._encoder_worker = None
        # CRITICAL: close the dataset's open parquet writers. lerobot v3 streams
        # episodes into a long-lived pq.ParquetWriter and only writes the closing
        # footer when finalize() is called; without this the data/episodes parquet
        # files are left un-finalized and are unreadable/invalid.
        self._finalize_dataset()

    def _finalize_dataset(self) -> None:
        """Flush + close the dataset's parquet writers so footers are written.

        Safe to call multiple times and on lerobot builds that predate
        ``finalize`` (older versions closed writers per episode).
        """
        with self._ds_lock:
            ds = self._dataset
            if ds is None:
                return
            finalize = getattr(ds, "finalize", None)
            if not callable(finalize):
                return
            try:
                finalize()
                _LOGGER.info("Dataset finalized (parquet writers closed).")
            except Exception as exc:
                _LOGGER.error("Dataset finalize failed: %s", exc)

    # ── episode control ─────────────────────────────────────────────────
    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frames_in_episode(self) -> int:
        """Number of frames captured in the current (or last) episode."""
        return self._frames_in_episode

    @property
    def is_encoding(self) -> bool:
        """True while one or more finished episodes are still encoding."""
        with self._encode_lock:
            return self._encode_pending > 0

    @property
    def state(self) -> str:
        """Coarse recorder state for the GUI: idle, recording or encoding.

        Recording takes priority: a new episode can be captured while the previous
        one is still encoding in the background (see ``is_encoding``).
        """
        if self._recording:
            return "recording"
        if self.is_encoding:
            return "encoding"
        return "idle"

    @property
    def recording_seconds(self) -> float:
        """Elapsed seconds of the current episode (live) or the last one."""
        if self._recording:
            return max(0.0, time.monotonic() - self._episode_start_t)
        return self._episode_elapsed

    @property
    def episodes(self) -> list[dict]:
        """History (most recent first), with live encode progress computed.

        ``save_episode`` is a blocking black box, so progress for an actively
        encoding episode is estimated from the elapsed time against an adaptive
        per-frame estimate. Queued-but-not-started saves report 0.0; finished
        saves report 1.0.
        """
        now = time.monotonic()
        with self._lock:
            out = []
            for entry in self._episodes:
                snapshot = dict(entry)
                status = snapshot.get("status")
                if status == "encoding":
                    start = snapshot.get("enc_start")
                    est = snapshot.get("enc_estimate", 0.0)
                    if start is None:
                        # Queued behind earlier saves (worker is serialized);
                        # surface as a distinct state so it doesn't look stuck at
                        # "encoding 0%".
                        snapshot["status"] = "queued"
                        snapshot["progress"] = 0.0
                    elif est <= 0:
                        snapshot["progress"] = 0.0
                    else:
                        snapshot["progress"] = min(0.97, (now - start) / est)
                elif status == "saved":
                    snapshot["progress"] = 1.0
                snapshot.pop("enc_start", None)
                snapshot.pop("enc_estimate", None)
                out.append(snapshot)
            return out

    def _mark_encode_started(self, eid: int | None, estimate: float) -> None:
        """Stamp the history entry with the moment its encode actually began."""
        if eid is None:
            return
        with self._lock:
            for entry in self._episodes:
                if entry.get("eid") == eid:
                    entry["enc_start"] = time.monotonic()
                    entry["enc_estimate"] = estimate
                    break

    def _record_history(
        self, status: str, frames: int, duration: float, index: int | None
    ) -> int:
        """Insert a history entry (most-recent-first) and return its unique id."""
        self._history_seq += 1
        eid = self._history_seq
        entry = {
            "eid": eid,
            "index": index,
            "status": status,  # encoding | saved | discarded | error
            "frames": frames,
            "duration": round(duration, 1),
            "fps": round(frames / duration, 1) if duration > 0 else 0.0,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        with self._lock:
            self._episodes.insert(0, entry)
            del self._episodes[50:]
        return eid

    def _update_history(self, eid: int, status: str, index: int | None) -> None:
        """Update an existing history entry in place (e.g. encoding -> saved)."""
        with self._lock:
            for entry in self._episodes:
                if entry.get("eid") == eid:
                    entry["status"] = status
                    if index is not None:
                        entry["index"] = index
                    entry["time"] = datetime.now().strftime("%H:%M:%S")
                    break

    def start_episode(self) -> None:
        if self._dataset is None:
            return
        # Capture is decoupled from the dataset: frames are buffered in RAM and
        # saved by the encoder worker off the capture path, so a new episode can
        # start immediately even while a previous one is still encoding.
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._frames_in_episode = 0
            self._live_records = []
            self._episode_start_t = time.monotonic()
            self._episode_elapsed = 0.0
        _LOGGER.info("Episode recording STARTED")

    def stop_episode(self) -> None:
        if self._dataset is None:
            return
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._episode_elapsed = max(0.0, time.monotonic() - self._episode_start_t)
            frames = self._frames_in_episode
            duration = self._episode_elapsed
            records = self._live_records
            self._live_records = []
        if len(records) >= self.min_frames:
            if self._defer_encoding:
                self._hold_raw(records, frames, duration)
            else:
                self._enqueue_save(records, frames, duration)
        else:
            # Too short to save: just release the buffered RAM and log it.
            with self._lock:
                self._buffered_frames = max(0, self._buffered_frames - len(records))
            _LOGGER.info(
                "Episode DISCARDED (too short, %d frames, not saved)", frames
            )
            self._record_history("discarded", frames, duration, None)

    # ── deferred encoding ───────────────────────────────────────────────
    @property
    def defer_encoding(self) -> bool:
        return self._defer_encoding

    def set_defer_encoding(self, enabled: bool) -> None:
        """Enable/disable deferred encoding for FUTURE stops.

        Already-held raw episodes are untouched; they wait for ``encode_pending``
        regardless of this flag.
        """
        self._defer_encoding = bool(enabled)
        _LOGGER.info("Deferred encoding %s", "ON" if enabled else "OFF")

    @property
    def pending_count(self) -> int:
        """Number of raw episodes captured but not yet sent to the encoder."""
        with self._lock:
            return len(self._pending_episodes)

    def _hold_raw(self, records: list[dict], frames: int, duration: float) -> None:
        """Hold a finished episode's frames in RAM in 'raw' state.

        The frames stay in memory (counted against the RAM budget) until
        ``encode_pending`` flushes them to the encoder. A history entry is shown
        immediately so the operator can see what is queued for later encoding.
        """
        eid = self._record_history("raw", frames, duration, None)
        with self._lock:
            self._pending_episodes.append(
                {
                    "records": records,
                    "frames": frames,
                    "duration": duration,
                    "eid": eid,
                }
            )
        _LOGGER.info("Episode HELD raw (%d frames) awaiting encode", frames)

    def encode_pending(self) -> int:
        """Flush all held 'raw' episodes into the encode queue, in order.

        Returns the number of episodes dispatched. Encoding still runs serially on
        the single worker (lerobot v2.1 requires contiguous indices), but the
        operator triggers it explicitly instead of it competing with recording.
        """
        with self._lock:
            pending = self._pending_episodes
            self._pending_episodes = []
        for job in pending:
            with self._encode_lock:
                self._encode_pending += 1
            # Flip the raw history entry to encoding so the queued/progress display
            # logic picks it up (it stays 'queued' until its turn).
            self._update_history(job["eid"], "encoding", None)
            self._encode_queue.put(job)
        if pending:
            _LOGGER.info("Encoding %d held episode(s)", len(pending))
        return len(pending)

    def _enqueue_save(
        self, records: list[dict], frames: int, duration: float
    ) -> None:
        """Queue a finished episode's buffered frames for background save.

        The captured frames live entirely in RAM (decoupled from the dataset), so
        queuing never touches ``self.episode_buffer`` and never blocks the
        sampling thread. The single encoder worker replays the frames into the
        dataset and calls ``save_episode()`` one episode at a time, preserving
        lerobot v2.1's contiguous-index requirement.
        """
        with self._encode_lock:
            self._encode_pending += 1
        # Show the episode immediately as "encoding" so it never vanishes from the
        # list between stop and save-complete; updated in place afterwards. The
        # on-disk index isn't known until the save runs (it equals
        # meta.total_episodes at that moment), so record it as None for now.
        eid = self._record_history("encoding", frames, duration, None)
        self._encode_queue.put(
            {
                "records": records,
                "frames": frames,
                "duration": duration,
                "eid": eid,
            }
        )

    def _encoder_loop(self) -> None:
        """Serialize episode saves so they persist in order, one at a time."""
        while True:
            job = self._encode_queue.get()
            try:
                if job is None:
                    return
                self._process_save(job)
            except Exception:
                # Log the exception type + full traceback: lerobot's internal
                # validation often raises a bare AssertionError with an empty
                # message, so ``str(exc)`` alone yields a useless "failed: " line.
                # ``logger.exception`` captures the stack so the cause of an ERROR
                # episode is actually diagnosable.
                _LOGGER.exception("save_episode failed")
                eid = job.get("eid") if job else None
                if eid is not None:
                    self._update_history(eid, "error", None)
                else:
                    self._record_history(
                        "error", job.get("frames", 0), job.get("duration", 0.0), None
                    )
            finally:
                if job is not None:
                    with self._encode_lock:
                        self._encode_pending -= 1
                self._encode_queue.task_done()

    def _process_save(self, job: dict) -> None:
        records = job["records"]
        frames = job["frames"]
        duration = job["duration"]
        eid = job.get("eid")
        # Arm the progress bar from the moment encoding actually starts.
        self._mark_encode_started(eid, max(0.5, frames * self._enc_spf))
        t0 = time.monotonic()
        # Replay the RAM-buffered frames into a fresh dataset buffer, then save.
        # Holding _ds_lock for the whole job keeps the dataset single-writer (only
        # this worker ever mutates it); the sampling thread writes to its own
        # _live_records list and is never blocked. lerobot v2.1 saves one complete
        # episode parquet + per-episode ffmpeg-encoded mp4 and asserts the on-disk
        # file counts match num_episodes, so the worker is serial.
        with self._ds_lock:
            ds = self._dataset
            idx = ds.meta.total_episodes  # contiguous index this save writes
            try:
                buf = getattr(ds, "episode_buffer", None)
                if buf is None or buf.get("size", 0):
                    # Start from a clean buffer at the correct index (a prior
                    # failed job may have left a partially filled one).
                    ds.episode_buffer = ds.create_episode_buffer(episode_index=idx)
                for record in records:
                    self._add_frame_safe(record)
                ds.save_episode()
            except Exception:
                # Saving failed; drop the partial buffer + staged images so a later
                # episode starts clean (indices realign to total_episodes).
                with contextlib.suppress(Exception):
                    self._delete_buffer_images(idx)
                    ds.episode_buffer = ds.create_episode_buffer(episode_index=idx)
                raise
        # Free the RAM held by this episode's buffered frames.
        with self._lock:
            self._buffered_frames = max(0, self._buffered_frames - len(records))
        records.clear()
        elapsed = time.monotonic() - t0
        if frames > 0 and elapsed > 0:
            # Refine the per-frame estimate (EMA) for the next episode's bar.
            self._enc_spf = 0.6 * self._enc_spf + 0.4 * (elapsed / frames)
        self.episode_count += 1
        _LOGGER.info("Episode SAVED (index %d, %d frames)", idx, frames)
        if eid is not None:
            self._update_history(eid, "saved", idx)
        else:
            self._record_history("saved", frames, duration, idx)

    def _delete_buffer_images(self, idx: int) -> None:
        """Remove the on-disk PNG frames staged for an episode index.

        Only ever called from the encoder worker (under ``_ds_lock``) to clean up a
        partially staged buffer after a failed save.
        """
        ds = self._dataset
        with contextlib.suppress(Exception):
            ds._wait_image_writer()
        try:
            enc = getattr(ds, "_streaming_encoder", None)
            if enc is not None:
                enc.cancel_episode()
        except Exception:
            pass
        for cam_id in self.camera_ids:
            key = f"observation.images.{cam_id}"
            try:
                get_dir = getattr(ds, "_get_image_file_dir", None)
                if callable(get_dir):
                    d = get_dir(idx, key)
                else:
                    # lerobot v2.1: derive the per-episode image dir from the
                    # frame-0 staged image path's parent.
                    d = ds._get_image_file_path(
                        episode_index=idx, image_key=key, frame_index=0
                    ).parent
                if d.is_dir():
                    shutil.rmtree(d)
            except Exception:
                pass

    def toggle_episode(self) -> None:
        if self._recording:
            self.stop_episode()
        else:
            self.start_episode()

    def discard_episode(self) -> None:
        """Stop the active episode and delete it WITHOUT saving."""
        if self._dataset is None:
            return
        with self._lock:
            if not self._recording:
                _LOGGER.info("Discard requested but not recording; ignored.")
                return
            self._recording = False
            self._episode_elapsed = max(0.0, time.monotonic() - self._episode_start_t)
            frames = self._frames_in_episode
            duration = self._episode_elapsed
            records = self._live_records
            self._live_records = []
            # Frames were only buffered in RAM (never written to the dataset), so
            # discarding is just dropping the list and freeing its budget.
            self._buffered_frames = max(0, self._buffered_frames - len(records))
        _LOGGER.info("Episode DISCARDED (user discard, %d frames not saved)", frames)
        self._record_history("discarded", frames, duration, None)

    # ── sampling thread ─────────────────────────────────────────────────
    def _run(self) -> None:
        period = 1.0 / float(self.fps)
        while not self._stop.is_set():
            t0 = time.monotonic()
            if self._recording:
                try:
                    self._sample_once()
                except Exception as exc:
                    _LOGGER.error("frame sample error: %s", exc)
            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _sample_once(self) -> None:
        frame = self.frame_provider()
        if frame is None:
            return
        state = np.asarray(frame.state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.dim:
            # Arm reader momentarily returned an incomplete state; skip this frame
            # rather than corrupt the dataset buffer with a bad shape.
            _LOGGER.debug(
                "skipping frame: state dim %d != expected %d",
                state.shape[0],
                self.dim,
            )
            return
        # Build a frame dict of RAM-OWNED copies. The buffered record outlives this
        # call (it's saved later by the encoder worker), so it must NOT alias the
        # camera's reusable latest-frame buffer or the provider's state array,
        # which are overwritten on the next sample.
        record: dict[str, object] = {
            "observation.state": np.array(state, dtype=np.float32),
        }
        for arm in self.arm_names:
            closed = bool(frame.grippers_closed.get(arm, False))
            record[f"observation.{arm}.gripper_is_closed"] = np.array(
                [closed], dtype=bool
            )
        for cam_id in self.camera_ids:
            img = self._normalize_image(frame.images.get(cam_id))
            record[f"observation.images.{cam_id}"] = np.array(img, dtype=np.uint8)
        with self._lock:
            if not self._recording:
                return
            if self._buffered_frames >= self._max_buffered_frames:
                # RAM safety valve hit (very long take or deep encode backlog):
                # drop this frame rather than risk OOM, warning at most ~once/2s.
                now = time.monotonic()
                if now - self._last_buf_warn > 2.0:
                    _LOGGER.warning(
                        "Live-capture RAM budget reached (%d frames); dropping "
                        "frames until the encode backlog drains.",
                        self._max_buffered_frames,
                    )
                    self._last_buf_warn = now
                return
            self._live_records.append(record)
            self._frames_in_episode += 1
            self._buffered_frames += 1

    def _add_frame_safe(self, record: dict[str, object]) -> None:
        """Add one buffered frame to the LeRobot dataset, rolling back a partial
        write. Called only from the encoder worker, which already holds
        ``_ds_lock``.

        lerobot's ``add_frame`` appends ``frame_index``/``timestamp``/``task`` and
        each feature column to ``episode_buffer`` and only increments ``size`` at
        the very end. If it raises partway through (e.g. while writing an image
        file), the buffer columns desync from ``size`` and ``save_episode`` later
        fails with a column-length mismatch. To stay resilient we snapshot the
        buffer lengths beforehand and restore them on failure so a single bad frame
        can't corrupt the whole episode.
        """
        buf = getattr(self._dataset, "episode_buffer", None)
        snapshot = None
        snap_size = None
        if buf is not None:
            snapshot = {k: len(v) for k, v in buf.items() if isinstance(v, list)}
            snap_size = buf.get("size")
        try:
            if self._task_as_arg:
                # lerobot v2.1: task is a separate argument, NOT a feature.
                self._dataset.add_frame(record, task=self.task)
            else:
                record["task"] = self.task
                self._dataset.add_frame(record)
        except Exception:
            buf = getattr(self._dataset, "episode_buffer", None)
            if buf is not None and snapshot is not None:
                for key, length in snapshot.items():
                    col = buf.get(key)
                    if isinstance(col, list) and len(col) > length:
                        del col[length:]
                if snap_size is not None:
                    buf["size"] = snap_size
            raise

    def _normalize_image(self, img: Any) -> np.ndarray:
        """Coerce a camera frame to a contiguous uint8 HxWx3 array of the
        configured resolution so a single odd frame can't break add_frame."""
        target = (self.image_h, self.image_w, 3)
        if img is None:
            return np.zeros(target, dtype=np.uint8)
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        if img.ndim != 3 or img.shape[2] != 3:
            return np.zeros(target, dtype=np.uint8)
        if img.shape[:2] != (self.image_h, self.image_w):
            if cv2 is not None:
                img = cv2.resize(
                    img, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA
                )
            else:
                return np.zeros(target, dtype=np.uint8)
        return np.ascontiguousarray(img, dtype=np.uint8)
