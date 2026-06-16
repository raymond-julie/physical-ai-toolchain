#!/usr/bin/env python3
"""LeRobot recorder node for the UR leader/follower teleop+record tool.

Drop-in replacement for ``recorder_node.py`` that records each episode as a
LeRobotDataset (parquet + mp4 videos) instead of a ROS 2 bag.

Episode lifecycle
-----------------
* ``/recorder/active`` rising edge  → begin a new in-memory episode buffer.
* While active                      → at ``fps`` Hz, the latest cached
                                       observation/action/image are appended
                                       as a frame via ``dataset.add_frame``.
* ``/recorder/active`` falling edge → ``dataset.save_episode()`` persists the
                                       episode to disk (parquet + encoded mp4).

Frame schema
------------
* ``observation.state``             — float32(7): 6 destination joints + gripper.
* ``action``                        — float32(7): 6 source joints + gripper.
* ``observation.images.color``      — RGB uint8 video frame.
* ``observation.images.depth``      — uint16 depth packed into 3-channel uint8.
* ``observation.gripper.is_closed`` — bool.

Requirements:  ROS 2 Humble, Python >= 3.10, ``pip install lerobot``,
``cv_bridge`` (``ros-humble-cv-bridge``).
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64

if TYPE_CHECKING:
    from cv_bridge import CvBridge
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
STATE_NAMES = [*JOINT_NAMES, "gripper"]
ACTION_NAMES = [f"target_{n}" for n in JOINT_NAMES] + ["target_gripper"]


class RecorderError(RuntimeError):
    """Raised when a required recorder dependency is unavailable."""


class EpisodeDecision(StrEnum):
    """Outcome of evaluating a finished episode before persisting it."""

    DISCARD_EMPTY = "discard_empty"
    DISCARD_SHORT = "discard_short"
    SAVE = "save"


def classify_episode(frames_in_episode: int, min_episode_frames: int) -> EpisodeDecision:
    """Decide whether a finished episode should be saved or discarded.

    Empty episodes are dropped silently; episodes shorter than
    ``min_episode_frames`` are treated as stray toggles and discarded.
    """
    if frames_in_episode == 0:
        return EpisodeDecision.DISCARD_EMPTY
    if frames_in_episode < min_episode_frames:
        return EpisodeDecision.DISCARD_SHORT
    return EpisodeDecision.SAVE


def reorder_joints(
    names: Sequence[str],
    positions: Sequence[float],
    joint_names: Sequence[str],
) -> np.ndarray | None:
    """Return ``positions`` reordered to ``joint_names``, or None when empty.

    When ``names`` does not contain the canonical joint names, the first
    ``len(joint_names)`` positions are returned as-is (assumed already ordered).
    """
    if not names or not positions:
        return None
    try:
        idx = [list(names).index(n) for n in joint_names]
    except ValueError:
        if len(positions) >= len(joint_names):
            return np.asarray(positions[: len(joint_names)], dtype=np.float32)
        return None
    return np.asarray([positions[i] for i in idx], dtype=np.float32)


def pack_depth_u16(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pack uint16 depth (mm) into (H, W, 3) uint8 with R=high byte, G=low byte."""
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    if depth.shape != (height, width):
        try:
            import cv2

            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        except ImportError:
            depth = depth[:height, :width]
    hi = (depth >> 8).astype(np.uint8)
    lo = (depth & 0xFF).astype(np.uint8)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[..., 0] = hi
    out[..., 1] = lo
    return out


def _import_cv_bridge() -> type[CvBridge]:
    try:
        from cv_bridge import CvBridge
    except ImportError as exc:  # pragma: no cover
        raise RecorderError(
            "cv_bridge missing. Install with: sudo apt install ros-humble-cv-bridge"
        ) from exc
    return CvBridge


def _import_lerobot_dataset() -> type[LeRobotDataset]:
    # Force HuggingFace Hub into offline mode BEFORE importing lerobot. Newer
    # lerobot versions call ``huggingface_hub.list_repo_refs(repo_id)`` during
    # dataset open, which 401s for purely local repo IDs like ``local/ur5_mirror``.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # pragma: no cover
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RecorderError("lerobot not installed. Install with: pip install --user lerobot") from exc
    return LeRobotDataset


class LeRobotRecorderNode(Node):
    """Bridges ROS 2 topics into LeRobotDataset episodes."""

    def __init__(self) -> None:
        super().__init__("lerobot_recorder_node")

        from rcl_interfaces.msg import ParameterDescriptor, ParameterType

        self.declare_parameter("repo_id", "local/ur5_mirror")
        self.declare_parameter("root", "./recordings_lerobot")
        # ``dynamic_typing=True`` so the launcher can pass ``15`` or ``15.0``
        # without ROS 2 Jazzy raising InvalidParameterTypeException.
        self.declare_parameter(
            "fps",
            30.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, dynamic_typing=True),
        )
        self.declare_parameter("task", "ur5_mirror_episode")
        self.declare_parameter("record_depth", True)
        self.declare_parameter("record_camera2", True)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("color_topic", "/camera1/camera1/color/image_raw")
        self.declare_parameter("depth_topic", "/camera1/camera1/depth/image_rect_raw")
        self.declare_parameter("color_topic2", "/camera2/camera2/color/image_raw")
        self.declare_parameter("depth_topic2", "/camera2/camera2/depth/image_rect_raw")
        self.declare_parameter("use_videos", True)
        # Discard episodes shorter than this many frames. Prevents a stray
        # DI0/GUI toggle from saving a 1-frame "ghost" episode.
        self.declare_parameter("min_episode_frames", 5)

        self.repo_id = self.get_parameter("repo_id").value
        self.root = Path(self.get_parameter("root").value).expanduser().resolve()
        self.fps = int(self.get_parameter("fps").value)
        self.task = self.get_parameter("task").value
        self.record_depth = bool(self.get_parameter("record_depth").value)
        self.record_camera2 = bool(self.get_parameter("record_camera2").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.color_topic2 = self.get_parameter("color_topic2").value
        self.depth_topic2 = self.get_parameter("depth_topic2").value
        self.use_videos = bool(self.get_parameter("use_videos").value)
        self.min_episode_frames = int(self.get_parameter("min_episode_frames").value)

        # Latest-message cache (filled by subscriber callbacks).
        self._lock = threading.Lock()
        self._latest_obs_joints: np.ndarray | None = None  # (6,) destination
        self._latest_action_joints: np.ndarray | None = None  # (6,) source
        self._latest_gripper_pos: float = 0.0
        self._latest_gripper_is_closed: bool = False
        self._latest_color: np.ndarray | None = None  # (H,W,3) uint8
        self._latest_depth: np.ndarray | None = None  # (H,W) uint16
        self._latest_color2: np.ndarray | None = None  # (H,W,3) uint8
        self._latest_depth2: np.ndarray | None = None  # (H,W) uint16

        self._bridge = _import_cv_bridge()()

        # Episode state.
        self.is_recording = False
        self.is_saving = False  # True while save_episode is running
        self.episode_count = 0
        self.frames_in_episode = 0

        self.dataset = self._open_or_create_dataset()

        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.create_subscription(Bool, "/recorder/active", self._active_cb, be_qos)
        # destination joints (observation.state)
        self.create_subscription(JointState, "/joint_states", self._dest_joints_cb, be_qos)
        # source joints (action)
        self.create_subscription(JointState, "/mirror/joint_states", self._src_joints_cb, be_qos)
        self.create_subscription(Float64, "/mirror/gripper/position", self._gripper_pos_cb, be_qos)
        self.create_subscription(Bool, "/mirror/gripper/is_closed", self._gripper_closed_cb, be_qos)
        self.create_subscription(Image, self.color_topic, self._color_cb, img_qos)
        if self.record_depth:
            self.create_subscription(Image, self.depth_topic, self._depth_cb, img_qos)
        if self.record_camera2:
            self.create_subscription(Image, self.color_topic2, self._color2_cb, img_qos)
            if self.record_depth:
                self.create_subscription(Image, self.depth_topic2, self._depth2_cb, img_qos)

        self._frame_period = 1.0 / float(self.fps)
        self.create_timer(self._frame_period, self._frame_tick)
        self.create_timer(5.0, self._log_tick)

        self.get_logger().info(
            f"LeRobotRecorder ready  repo_id={self.repo_id}  root={self.root}  "
            f"fps={self.fps}  depth={self.record_depth}"
        )

    def _features(self) -> dict[str, dict]:
        feats: dict[str, dict] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(STATE_NAMES),),
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(ACTION_NAMES),),
                "names": ACTION_NAMES,
            },
            "observation.gripper.is_closed": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["is_closed"],
            },
            "observation.images.color": {
                "dtype": "video" if self.use_videos else "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
            },
        }
        if self.record_depth:
            # LeRobot's image writer requires 3-channel uint8. Pack uint16 mm
            # depth losslessly into (R=high byte, G=low byte, B=0).
            feats["observation.images.depth"] = {
                "dtype": "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
            }
        if self.record_camera2:
            feats["observation.images.color2"] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
            }
            if self.record_depth:
                feats["observation.images.depth2"] = {
                    "dtype": "image",
                    "shape": (self.image_height, self.image_width, 3),
                    "names": ["height", "width", "channels"],
                }
        return feats

    def _open_or_create_dataset(self) -> LeRobotDataset:
        # Always create a fresh dataset per recorder session in a timestamped
        # subdirectory. This avoids two pitfalls of the installed lerobot:
        #   1. ``LeRobotDataset.create`` calls ``root.mkdir(exist_ok=False)``
        #      and crashes if any earlier run left the directory behind.
        #   2. Re-opening an existing dataset triggers a HuggingFace Hub lookup
        #      of the (fictitious) ``local/ur5_mirror`` repo, which 401s when
        #      the user is not authenticated.
        # Multiple DI0 episodes within one session still accumulate inside the
        # same dataset; only restarting the launcher creates a new one.
        lerobot_dataset_cls = _import_lerobot_dataset()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_root = self.root / f"session_{ts}"
        session_root.parent.mkdir(parents=True, exist_ok=True)

        # ``LeRobotDataset.create`` does the mkdir of ``session_root`` with
        # exist_ok=False, so make sure it does not exist yet.
        if session_root.exists():
            import shutil

            shutil.rmtree(session_root)

        self.get_logger().info(f"Creating new dataset @ {session_root}")
        create_kwargs = {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "root": session_root,
            "features": self._features(),
            "use_videos": self.use_videos,
        }
        try:
            # Flush meta/episodes/*.parquet after every save_episode so a
            # Ctrl+C exit can never lose unflushed metadata.
            return lerobot_dataset_cls.create(**create_kwargs, metadata_buffer_size=1)
        except TypeError as exc:
            if "metadata_buffer_size" not in str(exc):
                raise
            self.get_logger().warning(
                "lerobot install does not accept metadata_buffer_size kwarg; "
                "falling back to default buffering. A Ctrl+C during an unflushed "
                "batch may lose the last few episode-metadata records on this version."
            )
            return lerobot_dataset_cls.create(**create_kwargs)

    @staticmethod
    def _reorder_joints(msg: JointState) -> np.ndarray | None:
        """Return joints in canonical UR order, or None if mapping fails."""
        return reorder_joints(msg.name, msg.position, JOINT_NAMES)

    def _dest_joints_cb(self, msg: JointState) -> None:
        v = self._reorder_joints(msg)
        if v is not None:
            with self._lock:
                self._latest_obs_joints = v

    def _src_joints_cb(self, msg: JointState) -> None:
        v = self._reorder_joints(msg)
        if v is not None:
            with self._lock:
                self._latest_action_joints = v

    def _gripper_pos_cb(self, msg: Float64) -> None:
        with self._lock:
            self._latest_gripper_pos = float(msg.data)

    def _gripper_closed_cb(self, msg: Bool) -> None:
        with self._lock:
            self._latest_gripper_is_closed = bool(msg.data)

    def _color_cb(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            self.get_logger().warn(f"color decode failed: {exc}")
            return
        with self._lock:
            self._latest_color = img

    def _depth_cb(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if img.dtype != np.uint16:
                # normalize float depth (m) → uint16 (mm)
                img = np.clip(img * 1000.0, 0, 65535).astype(np.uint16)
        except Exception as exc:
            self.get_logger().warn(f"depth decode failed: {exc}")
            return
        with self._lock:
            self._latest_depth = img

    def _color2_cb(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            self.get_logger().warn(f"color2 decode failed: {exc}")
            return
        with self._lock:
            self._latest_color2 = img

    def _depth2_cb(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if img.dtype != np.uint16:
                img = np.clip(img * 1000.0, 0, 65535).astype(np.uint16)
        except Exception as exc:
            self.get_logger().warn(f"depth2 decode failed: {exc}")
            return
        with self._lock:
            self._latest_depth2 = img

    def _active_cb(self, msg: Bool) -> None:
        # Log every message, including no-ops, so unexpected episode splits can
        # be correlated with publisher activity.
        self.get_logger().info(
            f"/recorder/active rx data={msg.data} is_recording={self.is_recording} "
            f"is_saving={self.is_saving} frames_in_ep={self.frames_in_episode}"
        )
        # Ignore toggles that arrive while we are still flushing the previous
        # episode — those are almost always queued during the multi-second
        # ffmpeg encode and would create spurious episodes.
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
        self.get_logger().info(f'🔴 EPISODE {self.episode_count + 1} START — task="{self.task}"')

    def _stop_episode(self) -> None:
        self.is_recording = False
        decision = classify_episode(self.frames_in_episode, self.min_episode_frames)
        if decision == EpisodeDecision.DISCARD_EMPTY:
            self.get_logger().warn("Episode ended with 0 frames — discarded")
            return
        if decision == EpisodeDecision.DISCARD_SHORT:
            self.get_logger().warn(
                f"Episode ended with only {self.frames_in_episode} frames "
                f"(< min_episode_frames={self.min_episode_frames}) — discarding (likely stray toggle)"
            )
            try:
                self.dataset.clear_episode_buffer(delete_images=len(self.dataset.meta.image_keys) > 0)
            except Exception as exc:
                self.get_logger().error(f"clear_episode_buffer failed: {exc}")
            return
        self.is_saving = True
        try:
            self.dataset.save_episode()
            self.episode_count += 1
            self.get_logger().info(f"⏹  EPISODE {self.episode_count} SAVED — {self.frames_in_episode} frames")
        except Exception as exc:
            self.get_logger().error(f"save_episode failed: {exc}")
        finally:
            self.is_saving = False

    def _frame_tick(self) -> None:
        if not self.is_recording:
            return
        with self._lock:
            obs_j = self._latest_obs_joints
            act_j = self._latest_action_joints
            grip = self._latest_gripper_pos
            grip_closed = self._latest_gripper_is_closed
            color = self._latest_color
            depth = self._latest_depth
            color2 = self._latest_color2
            depth2 = self._latest_depth2

        # All required signals must have been received at least once.
        if obs_j is None or act_j is None or color is None:
            return
        if self.record_depth and depth is None:
            return
        if self.record_camera2 and color2 is None:
            return
        if self.record_camera2 and self.record_depth and depth2 is None:
            return

        obs_state = np.concatenate([obs_j, [np.float32(grip)]]).astype(np.float32)
        action = np.concatenate([act_j, [np.float32(grip)]]).astype(np.float32)

        color_f = self._fit_image(color, channels=3)
        frame = {
            "observation.state": obs_state,
            "action": action,
            "observation.gripper.is_closed": np.array([grip_closed], dtype=bool),
            "observation.images.color": color_f,
        }
        if self.record_depth:
            frame["observation.images.depth"] = self._pack_depth_u16(depth)
        if self.record_camera2:
            frame["observation.images.color2"] = self._fit_image(color2, channels=3)
            if self.record_depth:
                frame["observation.images.depth2"] = self._pack_depth_u16(depth2)

        try:
            self.dataset.add_frame(frame, task=self.task)
            self.frames_in_episode += 1
        except Exception as exc:
            self.get_logger().error(f"add_frame failed: {exc}")

    def _pack_depth_u16(self, depth: np.ndarray) -> np.ndarray:
        return pack_depth_u16(depth, self.image_height, self.image_width)

    def _fit_image(self, img: np.ndarray, channels: int, dtype: type = np.uint8) -> np.ndarray:
        """Resize / pad ``img`` to (image_height, image_width, channels)."""
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

    def _log_tick(self) -> None:
        status = "RECORDING" if self.is_recording else "IDLE"
        self.get_logger().info(
            f"[{status}] episodes={self.episode_count} frames_this_ep={self.frames_in_episode}"
        )

    def destroy_node(self) -> bool:
        if self.is_recording:
            self.get_logger().info("Shutdown — flushing active episode")
            self._stop_episode()
        # Explicitly finalize so meta/episodes parquet, stats, and info.json are
        # flushed before exit. ``.finalize()`` only exists on newer lerobot
        # releases; older versions auto-flush after every save_episode so the
        # call is a harmless no-op there.
        finalize = getattr(self.dataset, "finalize", None)
        if callable(finalize):
            try:
                finalize()
                self.get_logger().info("Dataset finalized cleanly")
            except Exception as exc:
                self.get_logger().error(f"dataset.finalize failed: {exc}")
        else:
            self.get_logger().debug(
                "lerobot install has no LeRobotDataset.finalize(); "
                "episodes auto-flushed at save time on this version"
            )
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LeRobotRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().info("Shutting down LeRobot Recorder Node")
    finally:
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
