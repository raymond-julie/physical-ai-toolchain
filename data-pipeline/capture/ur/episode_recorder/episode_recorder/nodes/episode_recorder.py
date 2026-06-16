#!/usr/bin/env python3
"""Generic N-Robot Episode Recorder Node — episode_recorder.

Writes each ``/recorder/active`` rising->falling edge as one episode in
a LeRobotDataset (parquet + mp4) under
``<root>/session_<timestamp>/``.

The recorder is vendor-agnostic. It subscribes to a configurable list
of robot namespaces and image topics; each robot is expected to
publish ``<ns>/joint_states``, ``<ns>/gripper/position``, and
``<ns>/gripper/is_closed`` (see :mod:`episode_recorder.nodes.robot_reader`).

Frame schema
------------
* ``observation.state`` — ``float32(N * (joints_per_robot + 1))``,
  blocked per robot as ``[joints..., gripper_position]``.
* ``action`` — copy of ``observation.state``. Provided because
  LeRobotDataset conventionally expects this key; with no commanded
  action available it is just the realised state at this frame.
* ``observation.<ns>.gripper_is_closed`` — bool per robot.
* ``observation.images.color_<i>`` — RGB uint8 (mp4 if ``use_videos``)
  for each entry of ``image_topics``.
* ``observation.images.depth_<i>`` — uint16 mm packed into 3-channel
  uint8 PNG for each entry of ``depth_topics`` (optional).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, String

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover
    print(
        "ERROR: cv_bridge missing. Install with:\n  sudo apt install ros-${ROS_DISTRO}-cv-bridge",
        file=sys.stderr,
    )
    raise

# HuggingFace Hub offline mode — local-only repo IDs would otherwise
# trigger an authenticated repo-refs lookup. Set before importing lerobot.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import lerobot.datasets.lerobot_dataset as _lerobot_dataset_mod
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover
    try:
        import lerobot.common.datasets.lerobot_dataset as _lerobot_dataset_mod
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print(
            "ERROR: lerobot not installed. Install with:\n  pip install --user lerobot",
            file=sys.stderr,
        )
        raise


def _patch_lerobot_video_codec(vcodec: str, pix_fmt: str = "yuv420p") -> None:
    """Override LeRobot's default ``libsvtav1`` PNG->MP4 encoder.

    LeRobot 0.3.x defaults to AV1 (``libsvtav1``) when encoding the
    per-episode PNG buffer into an mp4. The bitstream is valid (frames
    decode correctly via pyav/imageio) but most general-purpose players
    — Ubuntu Files / Totem preview, VS Code's video preview, ROS
    ``rqt_image_view``, older browsers and HuggingFace Hub's web
    preview — cannot decode AV1 in an mp4 container and render the
    clip as a fully black frame. Re-encoding with H.264 (``h264``)
    restores playback in every common viewer at roughly the same file
    size for these short episodes.

    Patches the ``encode_video_frames`` symbol that
    :meth:`LeRobotDataset.encode_episode_videos` calls, so the
    override applies to every ``save_episode()`` performed by this
    process.
    """
    original = _lerobot_dataset_mod.encode_video_frames

    def _wrapped(imgs_dir: Any, video_path: Any, fps: int, **kwargs: Any) -> Any:
        kwargs.setdefault("vcodec", vcodec)
        kwargs.setdefault("pix_fmt", pix_fmt)
        return original(imgs_dir, video_path, fps, **kwargs)

    _lerobot_dataset_mod.encode_video_frames = _wrapped


def _clean_list(values: list[Any] | None, *, default: list[str] | None = None) -> list[str]:
    """Strip whitespace and drop empty strings from a ROS string-array param."""
    if values is None:
        return list(default or [])
    return [str(v).strip() for v in values if str(v).strip()]


class EpisodeRecorderNode(Node):
    """N-robot, N-camera LeRobot dataset recorder."""

    def __init__(self) -> None:
        super().__init__("episode_recorder")

        # ── Parameters ──────────────────────────────────────────
        self.declare_parameter("repo_id", "local/dataset")
        self.declare_parameter("root", "./recordings_lerobot")
        self.declare_parameter("fps", 30)
        self.declare_parameter("task", "episode")

        # Robots
        self.declare_parameter("robot_namespaces", ["robot1", "robot2"])
        self.declare_parameter("joints_per_robot", 6)

        # Cameras
        self.declare_parameter(
            "image_topics",
            [
                "/camera1/camera1/color/image_raw",
                "/camera2/camera2/color/image_raw",
            ],
        )
        # depth_topics defaults to a single empty string so the rclpy
        # type inference treats it as ``string_array``; empty entries
        # are filtered out at runtime -> "no depth" by default.
        self.declare_parameter("depth_topics", [""])
        self.declare_parameter("image_height", 480)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("use_videos", True)
        self.declare_parameter("min_episode_frames", 5)
        # Video codec used when LeRobotDataset encodes the per-episode
        # PNG buffer to mp4. LeRobot's upstream default is
        # ``libsvtav1`` (AV1), whose bitstream is valid but plays back
        # as a black frame in most general-purpose viewers (Ubuntu
        # Files / Totem preview, VS Code, rqt_image_view, older
        # browsers). ``h264`` is universally decodable. Allowed
        # values: ``h264``, ``hevc``, ``libsvtav1``.
        self.declare_parameter("video_codec", "h264")

        self.repo_id = str(self.get_parameter("repo_id").value)
        self.root = Path(self.get_parameter("root").value).expanduser().resolve()
        self.fps = int(self.get_parameter("fps").value)
        self.task = str(self.get_parameter("task").value)
        self.namespaces = [
            ns.strip("/") for ns in _clean_list(self.get_parameter("robot_namespaces").value)
        ]
        if not self.namespaces:
            raise RuntimeError("robot_namespaces parameter is empty")
        self.j_per_r = int(self.get_parameter("joints_per_robot").value)
        self.image_topics = _clean_list(self.get_parameter("image_topics").value)
        self.depth_topics = _clean_list(self.get_parameter("depth_topics").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.use_videos = bool(self.get_parameter("use_videos").value)
        self.min_episode_frames = int(self.get_parameter("min_episode_frames").value)
        self.video_codec = str(self.get_parameter("video_codec").value).strip() or "h264"
        if self.use_videos:
            try:
                _patch_lerobot_video_codec(self.video_codec)
                self.get_logger().info(
                    f"mp4 encoder override active: vcodec={self.video_codec} "
                    "(overrides LeRobot default libsvtav1)"
                )
            except Exception as e:
                self.get_logger().warn(
                    f"Could not patch LeRobot video codec to "
                    f"{self.video_codec!r}: {e}. Falling back to "
                    "LeRobot default (libsvtav1 / AV1) which renders "
                    "black in many viewers."
                )
        # Roll over to a new session_<timestamp> directory after this
        # many successfully-saved episodes. 0 disables rollover (single
        # session for the lifetime of the process).
        self.declare_parameter("episodes_per_session", 10)
        self.episodes_per_session = int(self.get_parameter("episodes_per_session").value)
        # Per-image-topic warmup: at startup we wait up to this long
        # for each declared image topic to publish at least one frame.
        # Topics that never publish in time are dropped from the
        # dataset schema so a single dead camera doesn't block all
        # recordings (the frame tick requires every declared topic to
        # have produced a frame before it adds an entry).
        self.declare_parameter("image_topic_warmup_s", 20.0)
        self.image_topic_warmup_s = float(self.get_parameter("image_topic_warmup_s").value)

        # ── Latest-message cache ────────────────────────────────
        self._lock = threading.Lock()
        self._joints: dict[str, np.ndarray | None] = {ns: None for ns in self.namespaces}
        self._gpos: dict[str, float] = {ns: 0.0 for ns in self.namespaces}
        self._gclosed: dict[str, bool] = {ns: False for ns in self.namespaces}
        self._color: dict[str, np.ndarray | None] = {t: None for t in self.image_topics}
        self._depth: dict[str, np.ndarray | None] = {t: None for t in self.depth_topics}

        self._bridge = CvBridge()

        # ── Episode state ───────────────────────────────────────
        self.is_recording = False
        self.is_saving = False
        self.episode_count = 0
        self._episodes_in_session = 0
        self.frames_in_episode = 0
        # Serializes _frame_tick's add_frame() and _stop_episode's
        # save_episode(). Without this, a Ctrl+C arriving in the
        # middle of an active episode lets the executor's last
        # in-flight frame_tick race save_episode() — add_frame can
        # partially mutate the buffer (frame_index/timestamp/task
        # already appended; a feature column not yet appended) and
        # save_episode then crashes with
        #   "Column N named episode_index expected length X but got X-1"
        # -> no parquet, no mp4, only the buffered PNGs survive.
        self._dataset_lock = threading.Lock()
        # Diagnostic snapshot of which inputs were missing on the last
        # frame_tick. Populated only while recording. Used by
        # _log_tick so the operator sees WHY no frames are being
        # added (e.g. no joint_states from a robot whose Nova session
        # is not active).
        self._stall_missing_joints: list[str] = []
        self._stall_missing_color: list[str] = []
        self._stall_missing_depth: list[str] = []

        # ── QoS profiles ────────────────────────────────────────
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # /recorder/active uses TRANSIENT_LOCAL durability so a record
        # press published BEFORE this node finishes warmup (which
        # takes several seconds) is still delivered to us when we
        # subscribe. Without this, any click during warmup is
        # silently dropped and the recorder stays in IDLE forever.
        rel_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        # Callback groups so the MultiThreadedExecutor can run image
        # decode in parallel with state callbacks + frame_tick. Each
        # image topic gets its own MutuallyExclusiveCallbackGroup so
        # N cameras decode in parallel on N executor threads. All
        # state/control callbacks share one group.
        self._control_cbg = MutuallyExclusiveCallbackGroup()
        self._image_cbgs: dict[str, MutuallyExclusiveCallbackGroup] = {}

        # ── Subscriptions ───────────────────────────────────────
        # NOTE: the /recorder/active subscription is intentionally
        # NOT created here — it is registered later, AFTER warmup and
        # dataset creation, so a record button press during the
        # several-second warmup phase isn't silently swallowed as a
        # zero-frame episode (the frame_tick timer doesn't exist yet
        # during warmup, so any frames between start/stop would be
        # impossible to record).

        # Publisher for GUI status tile.
        self._stats_pub = self.create_publisher(String, "/recorder/stats", rel)

        for ns in self.namespaces:
            self.create_subscription(
                JointState,
                f"/{ns}/joint_states",
                self._make_joints_cb(ns),
                be,
                callback_group=self._control_cbg,
            )
            self.create_subscription(
                Float64,
                f"/{ns}/gripper/position",
                self._make_gpos_cb(ns),
                be,
                callback_group=self._control_cbg,
            )
            self.create_subscription(
                Bool,
                f"/{ns}/gripper/is_closed",
                self._make_gclosed_cb(ns),
                be,
                callback_group=self._control_cbg,
            )

        for t in self.image_topics:
            cbg = MutuallyExclusiveCallbackGroup()
            self._image_cbgs[t] = cbg
            self.create_subscription(Image, t, self._make_color_cb(t), img, callback_group=cbg)
        for t in self.depth_topics:
            cbg = MutuallyExclusiveCallbackGroup()
            self._image_cbgs[t] = cbg
            self.create_subscription(Image, t, self._make_depth_cb(t), img, callback_group=cbg)

        # ── Wait for image topics, drop dead ones ───────────────
        self._warmup_image_topics()

        # ── Build dataset (uses the now-pruned topic lists) ─────
        self.dataset = self._open_or_create_dataset()

        # ── Trigger subscription (now that we're ready) ─────────
        # TRANSIENT_LOCAL: catches any press published while we were
        # warming up.
        # Pin /recorder/active to the control callback group so a Stop
        # press is processed by a DIFFERENT executor thread than the
        # one running _frame_tick. Without this, _active_cb shares the
        # node's default Mutually-Exclusive group with the timers and
        # gets starved while _frame_tick is busy writing PNGs at fps Hz
        # — the Stop press can sit in the queue for many seconds.
        self.create_subscription(
            Bool,
            "/recorder/active",
            self._active_cb,
            rel_latched,
            callback_group=self._control_cbg,
        )

        # ── Timers ──────────────────────────────────────────────
        # Each timer gets its OWN MutuallyExclusiveCallbackGroup so the
        # MultiThreadedExecutor can run them on independent threads.
        # When they all share the default group (also MEx), _frame_tick
        # at fps Hz back-to-back starves _log_tick and _stats_tick —
        # the GUI's status tile then freezes (stats_age grows without
        # bound) even though recording is healthy.
        self._frame_tick_cbg = MutuallyExclusiveCallbackGroup()
        self._log_tick_cbg = MutuallyExclusiveCallbackGroup()
        self._stats_tick_cbg = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / self.fps, self._frame_tick, callback_group=self._frame_tick_cbg)
        self.create_timer(5.0, self._log_tick, callback_group=self._log_tick_cbg)
        self.create_timer(0.5, self._stats_tick, callback_group=self._stats_tick_cbg)

        self.get_logger().info(
            f"EpisodeRecorder ready: robots={self.namespaces} "
            f"images={len(self.image_topics)} depths={len(self.depth_topics)} "
            f"state_dim={len(self.namespaces) * (self.j_per_r + 1)} "
            f"fps={self.fps} root={self.root}"
        )

    def _warmup_image_topics(self) -> None:
        """Spin briefly so each declared image topic can deliver its
        first frame. Drop any topic that never publishes within
        ``image_topic_warmup_s`` so the recorder doesn't stall on a
        dead camera.
        """
        topics = list(self.image_topics) + list(self.depth_topics)
        if not topics or self.image_topic_warmup_s <= 0:
            return
        self.get_logger().info(
            f"Waiting up to {self.image_topic_warmup_s:.1f}s for image topics to publish: {topics}"
        )
        deadline = self.get_clock().now().nanoseconds + int(self.image_topic_warmup_s * 1e9)
        while True:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                all_color = all(self._color[t] is not None for t in self.image_topics)
                all_depth = all(self._depth[t] is not None for t in self.depth_topics)
            if all_color and all_depth:
                break
            if self.get_clock().now().nanoseconds >= deadline:
                break
        with self._lock:
            live_color = [t for t in self.image_topics if self._color[t] is not None]
            dead_color = [t for t in self.image_topics if self._color[t] is None]
            live_depth = [t for t in self.depth_topics if self._depth[t] is not None]
            dead_depth = [t for t in self.depth_topics if self._depth[t] is None]
        if dead_color:
            self.get_logger().warn(
                f"Image topics with no frames within warmup — dropping from dataset: {dead_color}"
            )
            self.image_topics = live_color
            with self._lock:
                for t in dead_color:
                    self._color.pop(t, None)
        if dead_depth:
            self.get_logger().warn(
                f"Depth topics with no frames within warmup — dropping from dataset: {dead_depth}"
            )
            self.depth_topics = live_depth
            with self._lock:
                for t in dead_depth:
                    self._depth.pop(t, None)
        if not self.image_topics:
            self.get_logger().error(
                "No image topics produced a frame during warmup. "
                "Recording will be unable to write frames."
            )

    # ── Features ────────────────────────────────────────────────

    def _state_names(self) -> list[str]:
        out: list[str] = []
        for ns in self.namespaces:
            for j in range(self.j_per_r):
                out.append(f"{ns}_joint_{j}")
            out.append(f"{ns}_gripper")
        return out

    def _features(self) -> dict:
        snames = self._state_names()
        feats: dict = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(snames),),
                "names": snames,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(snames),),
                "names": snames,
            },
        }
        for ns in self.namespaces:
            feats[f"observation.{ns}.gripper_is_closed"] = {
                "dtype": "bool",
                "shape": (1,),
                "names": ["is_closed"],
            }
        for i, _ in enumerate(self.image_topics):
            feats[f"observation.images.color_{i}"] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
            }
        for i, _ in enumerate(self.depth_topics):
            feats[f"observation.images.depth_{i}"] = {
                "dtype": "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
            }
        return feats

    def _open_or_create_dataset(self) -> LeRobotDataset:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_root = self.root / f"session_{ts}"
        session_root.parent.mkdir(parents=True, exist_ok=True)
        if session_root.exists():
            import shutil

            shutil.rmtree(session_root)
        self.get_logger().info(f"Creating new dataset @ {session_root}")
        self._session_dir = session_root
        return LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=session_root,
            features=self._features(),
            use_videos=self.use_videos,
        )

    def _rollover_session(self) -> None:
        """Finalize the current dataset and open a new session_<ts>.

        Caller must hold self._dataset_lock so an in-flight frame_tick
        can't add_frame to the dataset we're about to swap out.
        """
        prev_session = getattr(self, "_session_dir", None)
        prev_count = self._episodes_in_session
        # Finalize the outgoing dataset where supported (older LeRobot
        # releases); 0.3.x persists per save_episode so this is a no-op.
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            try:
                finalize()
            except Exception as e:
                self.get_logger().error(f"dataset.finalize on rollover failed: {e}")
        self.get_logger().info(
            f"Rolling over session after {prev_count} episodes (prev={prev_session})"
        )
        # Guarantee a distinct directory name even if rollover happens
        # within the same wall-clock second as session creation.
        time.sleep(1.0)
        self.dataset = self._open_or_create_dataset()
        self._episodes_in_session = 0

    # ── Subscriber factories ────────────────────────────────────

    def _make_joints_cb(self, ns: str) -> Callable[[JointState], None]:
        n = self.j_per_r

        def cb(msg: JointState) -> None:
            if not msg.position or len(msg.position) < n:
                return
            v = np.asarray(msg.position[:n], dtype=np.float32)
            with self._lock:
                self._joints[ns] = v

        return cb

    def _make_gpos_cb(self, ns: str) -> Callable[[Float64], None]:
        def cb(msg: Float64) -> None:
            with self._lock:
                self._gpos[ns] = float(msg.data)

        return cb

    def _make_gclosed_cb(self, ns: str) -> Callable[[Bool], None]:
        def cb(msg: Bool) -> None:
            with self._lock:
                self._gclosed[ns] = bool(msg.data)

        return cb

    def _make_color_cb(self, topic: str) -> Callable[[Image], None]:
        def cb(msg: Image) -> None:
            try:
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception as e:
                self.get_logger().warn(f"{topic} decode failed: {e}")
                return
            with self._lock:
                self._color[topic] = img

        return cb

    def _make_depth_cb(self, topic: str) -> Callable[[Image], None]:
        def cb(msg: Image) -> None:
            try:
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img.dtype != np.uint16:
                    img = np.clip(img * 1000.0, 0, 65535).astype(np.uint16)
            except Exception as e:
                self.get_logger().warn(f"{topic} decode failed: {e}")
                return
            with self._lock:
                self._depth[topic] = img

        return cb

    # ── Episode lifecycle ───────────────────────────────────────

    def _active_cb(self, msg: Bool) -> None:
        self.get_logger().info(
            f"/recorder/active rx data={msg.data} "
            f"is_recording={self.is_recording} is_saving={self.is_saving} "
            f"frames_in_ep={self.frames_in_episode}"
        )
        if self.is_saving:
            self.get_logger().warn("/recorder/active ignored — still saving previous episode")
            return
        if msg.data and not self.is_recording:
            self._start_episode()
        elif not msg.data and self.is_recording:
            self._stop_episode()

    def _start_episode(self) -> None:
        self.frames_in_episode = 0
        self.is_recording = True
        self.get_logger().info(
            f'🔴 EPISODE {self.episode_count + 1} START — task="{self.task}"'
        )

    def _stop_episode(self) -> None:
        self.is_recording = False
        if self.frames_in_episode == 0:
            stalls = []
            if self._stall_missing_joints:
                stalls.append("joints from " + ",".join(self._stall_missing_joints))
            if self._stall_missing_color:
                stalls.append("color from " + ",".join(self._stall_missing_color))
            if self._stall_missing_depth:
                stalls.append("depth from " + ",".join(self._stall_missing_depth))
            cause = " (cause: never received " + "; ".join(stalls) + ")" if stalls else ""
            self.get_logger().warn(
                f"Episode ended with 0 frames — discarded{cause}. "
                f"Dataset directory will contain only meta files until "
                f"an episode is actually recorded."
            )
            return
        if self.frames_in_episode < self.min_episode_frames:
            self.get_logger().warn(
                f"Episode ended with {self.frames_in_episode} frames < "
                f"min_episode_frames={self.min_episode_frames} — discarding"
            )
            try:
                with self._dataset_lock:
                    self.dataset.clear_episode_buffer(
                        delete_images=len(self.dataset.meta.image_keys) > 0
                    )
            except Exception as e:
                self.get_logger().error(f"clear_episode_buffer failed: {e}")
            return
        self.is_saving = True
        try:
            # Lock prevents a concurrent _frame_tick (running on
            # another executor thread) from calling add_frame while
            # save_episode is consuming the buffer.
            with self._dataset_lock:
                self.dataset.save_episode()
            self.episode_count += 1
            self._episodes_in_session += 1
            self.get_logger().info(
                f"⏹  EPISODE {self.episode_count} SAVED — {self.frames_in_episode} frames"
            )
            # Roll over to a fresh session after N saved episodes.
            if self.episodes_per_session > 0 and self._episodes_in_session >= self.episodes_per_session:
                try:
                    with self._dataset_lock:
                        self._rollover_session()
                except Exception as e:
                    self.get_logger().error(f"session rollover failed: {e}")
        except Exception as e:
            self.get_logger().error(f"save_episode failed: {e}")
        finally:
            self.is_saving = False

    # ── Frame sampling ──────────────────────────────────────────

    def _frame_tick(self) -> None:
        if not self.is_recording:
            return
        with self._lock:
            joints_snap = dict(self._joints)
            gpos_snap = dict(self._gpos)
            gclosed_snap = dict(self._gclosed)
            color_snap = dict(self._color)
            depth_snap = dict(self._depth)

        # All required topics must have produced at least one message.
        missing_joints = [ns for ns in self.namespaces if joints_snap[ns] is None]
        missing_color = [t for t in self.image_topics if color_snap[t] is None]
        missing_depth = [t for t in self.depth_topics if depth_snap[t] is None]
        if missing_joints or missing_color or missing_depth:
            self._stall_missing_joints = missing_joints
            self._stall_missing_color = missing_color
            self._stall_missing_depth = missing_depth
            return
        self._stall_missing_joints = []
        self._stall_missing_color = []
        self._stall_missing_depth = []

        parts: list[np.ndarray] = []
        for ns in self.namespaces:
            parts.append(joints_snap[ns])
            parts.append(np.array([gpos_snap[ns]], dtype=np.float32))
        state = np.concatenate(parts).astype(np.float32)

        frame: dict = {
            "observation.state": state,
            "action": state.copy(),
        }
        # NOTE: `task` is NOT a frame feature in LeRobotDataset — it
        # must be passed as a separate kwarg to `add_frame`. Including
        # it in the frame dict causes:
        #   "Feature mismatch in `frame` dictionary: Extra features: {'task'}"
        # and every frame gets rejected (-> 0-frame episodes, dataset
        # ends up with only meta files).
        for ns in self.namespaces:
            frame[f"observation.{ns}.gripper_is_closed"] = np.array(
                [gclosed_snap[ns]], dtype=bool
            )
        for i, t in enumerate(self.image_topics):
            frame[f"observation.images.color_{i}"] = self._fit_image(color_snap[t], 3)
        for i, t in enumerate(self.depth_topics):
            frame[f"observation.images.depth_{i}"] = self._pack_depth_u16(depth_snap[t])

        try:
            # LeRobotDataset >= 0.3 requires `task` as a positional/kwarg
            # argument on every add_frame call. (Older versions used
            # frame['task']; that path was removed upstream.)
            #
            # The lock serializes us against _stop_episode -> save_episode
            # so a Ctrl+C during recording cannot leave the episode_buffer
            # in a partially-appended state (frame_index/timestamp/task
            # appended but per-feature columns not yet appended -> parquet
            # build later fails with column-length mismatch).
            with self._dataset_lock:
                if not self.is_recording:
                    # Stop arrived while we were waiting for the lock; do
                    # not append after the buffer has been (or is being)
                    # saved.
                    return
                self.dataset.add_frame(frame, task=self.task)
                self.frames_in_episode += 1
        except Exception as e:
            self.get_logger().error(f"add_frame failed: {e}")

    # ── Image helpers ───────────────────────────────────────────

    def _pack_depth_u16(self, depth: np.ndarray) -> np.ndarray:
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16, copy=False)
        h, w = self.image_height, self.image_width
        if depth.shape != (h, w):
            try:
                import cv2

                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            except ImportError:
                depth = depth[:h, :w]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[..., 0] = (depth >> 8).astype(np.uint8)
        out[..., 1] = (depth & 0xFF).astype(np.uint8)
        return out

    def _fit_image(self, img: np.ndarray, channels: int, dtype: Any = np.uint8) -> np.ndarray:
        h, w = self.image_height, self.image_width
        if img.shape[0] != h or img.shape[1] != w:
            try:
                import cv2

                interp = cv2.INTER_NEAREST if dtype == np.uint16 else cv2.INTER_AREA
                img = cv2.resize(img, (w, h), interpolation=interp)
            except ImportError:
                img = img[:h, :w]
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[2] != channels:
            img = img[..., :channels]
        return img.astype(dtype, copy=False)

    # ── Status ──────────────────────────────────────────────────

    def _log_tick(self) -> None:
        status = "RECORDING" if self.is_recording else "IDLE"
        extra = ""
        if self.is_recording and self.frames_in_episode == 0:
            stalls = []
            if self._stall_missing_joints:
                stalls.append("no joint_states from " + ",".join(self._stall_missing_joints))
            if self._stall_missing_color:
                stalls.append("no color frames from " + ",".join(self._stall_missing_color))
            if self._stall_missing_depth:
                stalls.append("no depth frames from " + ",".join(self._stall_missing_depth))
            if stalls:
                extra = (
                    " — STALLED: "
                    + "; ".join(stalls)
                    + ". Episode will be discarded on Stop unless inputs start flowing."
                )
        self.get_logger().info(
            f"[{status}] episodes={self.episode_count} "
            f"frames_this_ep={self.frames_in_episode}{extra}"
        )

    def _stats_tick(self) -> None:
        """Publish a JSON status snapshot on /recorder/stats for the GUI."""
        if self._stats_pub is None:
            return
        payload = {
            "is_recording": bool(self.is_recording),
            "is_saving": bool(self.is_saving),
            "episode_count": int(self.episode_count),
            "frames_in_episode": int(self.frames_in_episode),
            "fps": int(self.fps),
            "dataset_root": str(self.root),
            "session_dir": str(getattr(self, "_session_dir", "") or ""),
            "namespaces": list(self.namespaces),
            "num_image_topics": len(self.image_topics),
            "num_depth_topics": len(self.depth_topics),
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._stats_pub.publish(msg)

    # ── Cleanup ─────────────────────────────────────────────────

    def destroy_node(self) -> None:
        # Episode flushing has been moved to main()'s finally block so
        # it runs BEFORE executor.shutdown() — running save_episode here
        # would race a still-in-flight frame_tick on another executor
        # thread and crash with a column-length mismatch.
        # `finalize` existed in older LeRobot releases; in 0.3.x episodes
        # are persisted per `save_episode()` call, so there is nothing to
        # flush at shutdown. Call it only if the installed version still
        # exposes it.
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            try:
                finalize()
                self.get_logger().info("Dataset finalized cleanly")
            except Exception as e:
                self.get_logger().error(f"dataset.finalize failed: {e}")
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = EpisodeRecorderNode()
    # MultiThreadedExecutor so image decode for N cameras + the
    # frame_tick + state callbacks can run in parallel. Without this
    # the single ROS thread is dominated by image deserialization
    # under load and frame_tick misses its 1/fps deadline.
    #
    # Thread budget: N image topics (one MEx group each, one thread
    # per group when busy) + control (joints + /recorder/active) +
    # frame_tick + log_tick + stats_tick = N + 4 distinct groups that
    # can run concurrently. Add a few extra threads so service/timer
    # bookkeeping isn't starved when all groups are simultaneously
    # active.
    n_streams = max(1, len(node.image_topics) + len(node.depth_topics))
    executor = MultiThreadedExecutor(num_threads=n_streams + 6)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().info("Shutting down EpisodeRecorder")
    finally:
        # Flush the active episode BEFORE tearing down the executor and
        # ROS context. Doing it after executor.shutdown() races the last
        # in-flight frame_tick callback and runs save_episode against a
        # context whose publishers are already invalid — yielding
        # 322/323-length column mismatches and no parquet on disk.
        if node.is_recording:
            try:
                node.get_logger().info("Shutdown — flushing active episode")
                node._stop_episode()
            except Exception as e:
                with contextlib.suppress(Exception):
                    node.get_logger().error(f"shutdown flush failed: {e}")
        executor.shutdown()
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
