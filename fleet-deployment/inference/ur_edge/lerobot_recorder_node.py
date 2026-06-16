#!/usr/bin/env python3
"""LeRobot recorder node for the UR edge runtime.

Records each episode as a LeRobotDataset (parquet + mp4 videos).

Episode lifecycle
-----------------
* ``/recorder/active`` rising edge  → begin a new in-memory episode buffer.
* While active                      → at ``fps`` Hz, the latest cached
                                       observation/action/image are appended
                                       as a frame via ``dataset.add_frame``.
* ``/recorder/active`` falling edge → ``dataset.save_episode(task=...)`` is
                                       called, persisting the episode to disk
                                       (parquet + encoded mp4 videos).

Frame schema
------------
* ``observation.state``           — float32(7): destination 6 joints + gripper
                                    analog position (``[j1..j6, gripper]``).
* ``observation.tcp_pose``        — float32(7): destination TCP pose in base
                                    frame + gripper bit ``[x,y,z,rx,ry,rz,
                                    gripper]`` (m / axis-angle rad / bool). Kept
                                    for human-readable playback / Isaac Sim / UR
                                    driver replay.
* ``observation.tcp_pose_rot6d``  — float32(10): destination TCP pose using 6D
                                    continuous rotation representation + gripper
                                    bit ``[x,y,z, R00,R10,R20,R01,R11,R21,
                                    gripper]``. No singularities or wraparound;
                                    preferred for TCP-action training (GR00T /
                                    pi0-EE / OpenVLA).
* ``observation.gripper.is_closed`` — bool: destination gripper closed bit.
* ``observation.gripper.position`` — float32(1): destination Robotiq encoder
                                    feedback [0,1] (standalone analog; not in
                                    joint/TCP vectors).
* ``action``                      — float32(7): source 6 joints + gripper bit
                                    (``[j1..j6, gripper]``), LeRobot canonical
                                    joint-action.
* ``action.joints``               — float32(7): same data as ``action``.
* ``action.tcp_pose``             — float32(7): source TCP pose + gripper bit
                                    (alternate task-space action for human/Isaac
                                    playback).
* ``action.tcp_pose_rot6d``       — float32(10): source TCP pose with rot6d
                                    rotation + gripper bit (alternate task-space
                                    action for GR00T / pi0-EE / OpenVLA training).
* ``action.gripper``              — bool: source gripper closed bit.
* ``action.gripper.position``     — float32(1): source Robotiq encoder feedback
                                    [0,1] (standalone analog; not in joint/TCP
                                    vectors).
* ``observation.images.scene``    — RGB uint8 video frame (third-person).
* ``observation.images.wrist``    — RGB uint8 video frame (eye-in-hand).
* ``observation.images.<role>_depth`` — uint16 depth packed as 3-ch PNG.
                                    Disabled when ``record_depth`` is False.

Requirements:  ROS 2 Humble, Python ≥ 3.10, ``pip install lerobot``,
``cv_bridge`` (``ros-humble-cv-bridge``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover
    CvBridge = None

# Force HuggingFace Hub into offline mode BEFORE importing lerobot. Newer lerobot
# versions call ``huggingface_hub.list_repo_refs(repo_id)`` during dataset open,
# which 401s for purely local repo IDs like ``local/ur10_mirror``.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
except ImportError:  # pragma: no cover
    try:
        # older lerobot layout
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
    except ImportError:
        raise

_LOGGER = logging.getLogger(__name__)

if CvBridge is None:  # pragma: no cover
    _LOGGER.warning("cv_bridge missing — falling back to manual decode")


# ── Manual sensor_msgs/Image decoders ──
# cv_bridge's C extension is built against numpy 1.x. Under numpy 2.x it raises
# `AttributeError: _ARRAY_API not found` on every imgmsg_to_cv2() call, which
# would silently kill the recorder's image callbacks. These small helpers cover
# the encodings the realsense driver actually publishes (rgb8 / bgr8 / 16UC1 /
# 32FC1) without involving cv_bridge.

def _decode_color_msg(msg: Image) -> np.ndarray | None:
    enc = msg.encoding
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, 3)
    if enc == "rgb8":
        return arr
    if enc == "bgr8":
        return arr[..., ::-1].copy()  # BGR -> RGB
    return None


def _decode_depth_msg(msg: Image) -> np.ndarray | None:
    enc = msg.encoding
    if enc in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width)
    if enc == "32FC1":
        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width)
        return np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    return None


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
STATE_NAMES = JOINT_NAMES + ["gripper"]
ACTION_NAMES = [f"target_{n}" for n in STATE_NAMES]
TCP_POSE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
# Every joint- and TCP-flavoured vector in the schema is suffixed with the
# gripper bit (binary 0/1) so the column ordering is consistent:
#   joints  vector: [j1..j6, gripper]            → len 7
#   tcp_pose vector:[x,y,z,rx,ry,rz, gripper]    → len 7
TCP_POSE_NAMES_WITH_GRIPPER = TCP_POSE_NAMES + ["gripper"]
# rot6d task-space vector layout (Zhou et al. 2019, ``On the Continuity of
# Rotation Representations in Neural Networks``). The 6 floats are the first two
# columns of the rotation matrix flattened in column-major order: [R00, R10,
# R20, R01, R11, R21]. Decoder reconstructs the third column via Gram-Schmidt +
# cross product. This representation has no singularities and no wraparound,
# making it the standard choice for GR00T / pi0-EE / OpenVLA TCP-action heads.
TCP_POSE_ROT6D_NAMES = [
    "tcp_x", "tcp_y", "tcp_z",
    "tcp_rot6d_0", "tcp_rot6d_1", "tcp_rot6d_2",
    "tcp_rot6d_3", "tcp_rot6d_4", "tcp_rot6d_5",
    "gripper",
]


def _axis_angle_to_rot6d(rvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector (axis*angle, rad) to a rot6d vector.

    rot6d is the first two columns of the rotation matrix, flattened in
    column-major order to 6 floats. The third column is recoverable at decode
    time via Gram-Schmidt + cross product, so the 6-vector fully determines the
    rotation while remaining continuous everywhere (unlike Euler / axis-angle,
    which wrap at +-pi).
    """
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        # Near identity: Taylor expansion -> R approx I + skew(r).
        K = np.array([[0.0, -r[2], r[1]],
                      [r[2], 0.0, -r[0]],
                      [-r[1], r[0], 0.0]])
        R = np.eye(3) + K
    else:
        axis = r / theta
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        # Rodrigues' formula.
        R = (np.eye(3)
             + np.sin(theta) * K
             + (1.0 - np.cos(theta)) * (K @ K))
    # Column-major flatten of first two columns: [R[:,0], R[:,1]].
    return R[:, :2].T.reshape(6).astype(np.float32)


class LeRobotRecorderNode(Node):
    """Bridges ROS 2 topics → LeRobotDataset episodes."""

    def __init__(self) -> None:
        super().__init__("lerobot_recorder_node")

        # ── Parameters ───────────────────────────────────────
        self.declare_parameter("repo_id", "local/ur10_mirror")
        self.declare_parameter("root", "./recordings_lerobot")
        self.declare_parameter("fps", 30)
        self.declare_parameter("task", "Pick Large White Gear, Place Blue Bin")
        # Robot make/model written into meta/info.json. LeRobot leaves this null
        # by default; setting it lets downstream tooling and HuggingFace dataset
        # cards identify the embodiment.
        self.declare_parameter("robot_type", "UR10_Single")
        self.declare_parameter("record_depth", True)
        self.declare_parameter("record_camera2", True)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("color_topic",
                               "/camera1/camera1/color/image_raw")
        self.declare_parameter("depth_topic",
                               "/camera1/camera1/depth/image_rect_raw")
        self.declare_parameter("color_topic2",
                               "/camera2/camera2/color/image_raw")
        self.declare_parameter("depth_topic2",
                               "/camera2/camera2/depth/image_rect_raw")
        # Camera role labels — used to name the dataset feature keys so the
        # LeRobot 3.0 single-arm convention is followed:
        #   observation.images.<role>   (e.g. ``scene``, ``wrist``)
        # Default: camera1 = scene (third-person), camera2 = wrist (eye-in-hand
        # on the destination tool-flange). Flip via params if cameras are
        # swapped.
        self.declare_parameter("camera1_role", "scene")
        self.declare_parameter("camera2_role", "wrist")
        self.declare_parameter("use_videos", True)
        # Discard episodes shorter than this many frames. Prevents a stray
        # DI0/GUI toggle from saving a 1-frame "ghost" episode.
        self.declare_parameter("min_episode_frames", 5)

        self.repo_id = self.get_parameter("repo_id").value
        self.root = Path(self.get_parameter("root").value).expanduser().resolve()
        self.fps = int(self.get_parameter("fps").value)
        self.task = self.get_parameter("task").value
        self.robot_type = self.get_parameter("robot_type").value or None
        self.record_depth = bool(self.get_parameter("record_depth").value)
        self.record_camera2 = bool(self.get_parameter("record_camera2").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.color_topic2 = self.get_parameter("color_topic2").value
        self.depth_topic2 = self.get_parameter("depth_topic2").value
        self.camera1_role = str(
            self.get_parameter("camera1_role").value).strip() or "scene"
        self.camera2_role = str(
            self.get_parameter("camera2_role").value).strip() or "wrist"
        self.use_videos = bool(self.get_parameter("use_videos").value)
        self.min_episode_frames = int(
            self.get_parameter("min_episode_frames").value)

        # ── Latest-message cache (filled by subscriber callbacks) ─
        # Naming convention:
        #   _obs_*  → destination robot (achieved / measured) → observation.*
        #   _act_*  → source robot (commanded by leader)       → action.*
        self._lock = threading.Lock()
        # Serializes mutating calls on ``self.dataset`` (``add_frame``,
        # ``save_episode``, ``clear_episode_buffer``). The recorder runs on a
        # MultiThreadedExecutor with a ReentrantCallbackGroup so a timer-driven
        # ``_frame_tick`` can race with a stop edge that calls ``save_episode``
        # — corrupting ``episode_buffer`` and eventually triggering ``KeyError:
        # 'size'`` on every subsequent frame. Holding ``_dataset_lock`` around
        # every dataset mutation makes those calls mutually exclusive.
        self._dataset_lock = threading.Lock()
        # Latched True when ``add_frame`` raises mid-episode. The buffer is then
        # in an inconsistent column-length state; we discard the episode at stop
        # time instead of attempting save_episode.
        self._episode_corrupt = False
        # Observation (destination): joints, TCP, gripper analog + bit
        self._obs_joints: np.ndarray | None = None       # (6,) dest
        self._obs_tcp_pose: np.ndarray | None = None     # (6,) dest
        self._obs_gripper_pos: float = 0.0               # analog [0,1] dest
        self._obs_gripper_closed: bool = False           # bit dest
        # Action (source): joints, TCP, gripper bit
        self._act_joints: np.ndarray | None = None       # (6,) source
        self._act_tcp_pose: np.ndarray | None = None     # (6,) source
        self._act_gripper_pos: float = 0.0               # analog [0,1] source
        self._act_gripper_closed: bool = False           # bit source
        # Cameras
        self._latest_color: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_color2: np.ndarray | None = None
        self._latest_depth2: np.ndarray | None = None

        self._bridge = None  # cv_bridge bypassed (numpy 2.x compat); see _decode_*_msg

        # ── Episode state ─────────────────────────────────────
        self.is_recording = False
        self.is_saving = False  # True while save_episode is running
        # Stash the most recent /recorder/active value received while a save was
        # in flight, so we can apply it the moment the save finishes. Otherwise
        # the user mashing Start during the ~15 s ffmpeg encode would silently
        # drop the next episode.
        self._pending_active: bool | None = None
        self.episode_count = 0
        self.frames_in_episode = 0

        # ── Build / load LeRobotDataset ───────────────────────
        self.dataset = self._open_or_create_dataset()

        # ── ROS QoS profiles ──────────────────────────────────
        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=2,
        )

        # ReentrantCallbackGroup so the MultiThreadedExecutor can run
        # image-decode callbacks in parallel with the 15 Hz frame timer and
        # joint/gripper/tcp callbacks. Without this every callback on this node
        # would still serialize on a single thread (the default
        # MutuallyExclusiveCallbackGroup), which is what was starving the timer
        # down to ~6 fps. Concurrent access to the ``self._latest_*`` cache is
        # already protected by ``self._lock``.
        cb_group = ReentrantCallbackGroup()

        # ── Subscriptions ─────────────────────────────────────
        self.create_subscription(Bool, "/recorder/active",
                                 self._active_cb, be_qos,
                                 callback_group=cb_group)
        # Latched ``/recorder/saving`` so the GUI can grey out Start while
        # save_episode() is flushing parquet + mp4. Otherwise the user has no
        # indication that pressing Start during the 10-20 s encode is queued
        # instead of immediate.
        saving_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._saving_pub = self.create_publisher(
            Bool, "/recorder/saving", saving_qos)
        # Publish an initial False so a late-joining GUI doesn't see a stale
        # latched True from a prior run.
        self._publish_saving(False)
        # New-session pulse: rotate to a fresh timestamped dataset folder without
        # restarting the node. Published by the GUI.
        self.create_subscription(Bool, "/recorder/new_session",
                                 self._new_session_cb, be_qos,
                                 callback_group=cb_group)
        # Live task-description updates from the GUI. The string set here is
        # attached to every subsequent add_frame() and therefore tags the
        # episode at save_episode() time. Used by SmolVLA / pi0 / GR00T as the
        # language instruction.
        self.create_subscription(String, "/recorder/task_description",
                                 self._task_description_cb, be_qos,
                                 callback_group=cb_group)
        # Live robot-type label updates from the GUI. The string is baked into
        # ``info.json`` and the session folder name on the *next* dataset
        # rotation (new session pulse or recorder restart). Refused mid-recording
        # so an active episode keeps a stable robot_type field.
        self.create_subscription(String, "/recorder/robot_type",
                                 self._robot_type_cb, be_qos,
                                 callback_group=cb_group)

        # ── Observation subscriptions (DESTINATION robot) ──
        # observation.state[0:6] : destination joints achieved
        self.create_subscription(JointState, "/destination/joint_states",
                                 self._obs_joints_cb, be_qos,
                                 callback_group=cb_group)
        # observation.tcp_pose : destination TCP achieved
        self.create_subscription(Float64MultiArray, "/destination/tcp_pose",
                                 self._obs_tcp_pose_cb, be_qos,
                                 callback_group=cb_group)
        # observation.state[6] (analog) : destination gripper position
        self.create_subscription(Float64, "/destination/gripper/position",
                                 self._obs_gripper_pos_cb, be_qos,
                                 callback_group=cb_group)
        # observation.gripper.is_closed : destination gripper bit
        self.create_subscription(Bool, "/destination/gripper/is_closed",
                                 self._obs_gripper_closed_cb, be_qos,
                                 callback_group=cb_group)

        # ── Action subscriptions (SOURCE robot → leader command) ──
        # action.joints : source joints (leader command)
        self.create_subscription(JointState, "/mirror/joint_states",
                                 self._act_joints_cb, be_qos,
                                 callback_group=cb_group)
        # action.tcp_pose : source TCP (leader command in task space)
        self.create_subscription(Float64MultiArray, "/mirror/tcp_pose",
                                 self._act_tcp_pose_cb, be_qos,
                                 callback_group=cb_group)
        # action.gripper (analog + bit) : source gripper command
        self.create_subscription(Float64, "/mirror/gripper/position",
                                 self._act_gripper_pos_cb, be_qos,
                                 callback_group=cb_group)
        self.create_subscription(Bool, "/mirror/gripper/is_closed",
                                 self._act_gripper_closed_cb, be_qos,
                                 callback_group=cb_group)

        # ── Cameras ──
        self.create_subscription(Image, self.color_topic,
                                 self._color_cb, img_qos,
                                 callback_group=cb_group)
        if self.record_depth:
            self.create_subscription(Image, self.depth_topic,
                                     self._depth_cb, img_qos,
                                     callback_group=cb_group)
        if self.record_camera2:
            self.create_subscription(Image, self.color_topic2,
                                     self._color2_cb, img_qos,
                                     callback_group=cb_group)
            if self.record_depth:
                self.create_subscription(Image, self.depth_topic2,
                                         self._depth2_cb, img_qos,
                                         callback_group=cb_group)

        # ── Frame sampling timer ──────────────────────────────
        self._frame_period = 1.0 / float(self.fps)
        self.create_timer(self._frame_period, self._frame_tick,
                          callback_group=cb_group)

        # 1 Hz status log
        self.create_timer(5.0, self._log_tick,
                          callback_group=cb_group)

        self.get_logger().info(
            f"LeRobotRecorder ready  repo_id={self.repo_id}  "
            f"root={self.root}  fps={self.fps}  "
            f"depth={self.record_depth}")

    # ── Dataset construction ─────────────────────────────────

    def _features(self) -> dict:
        # Schema summary:
        #   observation.*           → DESTINATION robot (achieved / measured)
        #   action / action.joints  → SOURCE robot joint command (canonical)
        #   action.tcp_pose         → SOURCE robot TCP   command (alternate)
        #
        # The canonical ``action`` column always carries 6 joints + gripper bit
        # (the LeRobot canonical for single-arm). Anyone who wants to train on
        # TCP-action instead simply reads ``action.tcp_pose`` (already populated
        # every frame) and the GR00T ``modality.json`` describes both options.
        feats = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(STATE_NAMES),),
                "names": STATE_NAMES,
                "info": {
                    "representation": "joint_position",
                    "unit": "rad (joints), bool (gripper)",
                    "layout": "[j1, j2, j3, j4, j5, j6, gripper]",
                    "source": ("follower/destination UR10e measured joints "
                               "+ gripper closed bit (0.0/1.0)"),
                },
            },
            "observation.tcp_pose_rot6d": {
                "dtype": "float32",
                "shape": (len(TCP_POSE_ROT6D_NAMES),),
                "names": TCP_POSE_ROT6D_NAMES,
                "info": {
                    "representation": "tcp_pose_rot6d",
                    "layout": ("[x, y, z, R00, R10, R20, R01, R11, R21,"
                               " gripper]"),
                    "unit": ("m (xyz), unitless (rot6d, first two cols "
                             "of rotation matrix column-major), "
                             "bool (gripper)"),
                    "frame": "robot_base",
                    "source": ("follower/destination UR10e RTDE "
                               "getActualTCPPose rotation vector "
                               "converted via Rodrigues"),
                    "note": ("Continuous rotation representation "
                             "(Zhou et al. 2019). Recommended for "
                             "GR00T / pi0-EE / OpenVLA TCP-action heads."),
                },
            },
            "observation.tcp_pose": {
                "dtype": "float32",
                "shape": (len(TCP_POSE_NAMES_WITH_GRIPPER),),
                "names": TCP_POSE_NAMES_WITH_GRIPPER,
                "info": {
                    "representation": "tcp_pose",
                    "layout": "[x, y, z, rx, ry, rz, gripper]",
                    "unit": "m (xyz), rad axis-angle (rxryrz), bool (gripper)",
                    "frame": "robot_base",
                    "source": ("follower/destination UR10e RTDE "
                               "getActualTCPPose + gripper closed bit"),
                },
            },
            "observation.gripper.is_closed": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["is_closed"],
            },
            "observation.gripper.position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["position"],
                "info": {
                    "representation": "gripper_analog",
                    "unit": "[0,1] Robotiq encoder feedback (pos/255)",
                    "source": ("follower/destination Robotiq encoder "
                               "feedback. Standalone column — NOT mirrored "
                               "into ``observation.state[6]`` (which holds "
                               "the discrete closed bit)."),
                    "note": ("Physics-affected jaw position; never reaches "
                             "1.0 even when fully clamped. Companion to "
                             "``observation.gripper.is_closed``."),
                },
            },
            "action": {
                "dtype": "float32",
                "shape": (len(ACTION_NAMES),),
                "names": ACTION_NAMES,
                "info": {
                    "representation": "joint_position",
                    "unit": "rad (joints), bool (gripper)",
                    "layout": "[j1, j2, j3, j4, j5, j6, gripper]",
                    "source": ("leader/source UR10e measured joints + gripper "
                               "closed bit (0.0/1.0)"),
                    "note": ("Canonical LeRobot 3.0 single-arm action: "
                             "follower-equivalent joint command. To train "
                             "on TCP-action instead, read "
                             "``action.tcp_pose`` and rewire modality.json."),
                },
            },
            # Extension action columns — always recorded. Same [j1..j6, gripper]
            # / [x,y,z,rx,ry,rz, gripper] layout as the canonical ``action``
            # column.
            "action.joints": {
                "dtype": "float32",
                "shape": (len(STATE_NAMES),),
                "names": [f"target_{n}" for n in STATE_NAMES],
                "info": {
                    "representation": "joint_position",
                    "duplicate_of": "action",
                    "note": "Same data as ``action``, kept for clarity.",
                },
            },
            "action.tcp_pose": {
                "dtype": "float32",
                "shape": (len(TCP_POSE_NAMES_WITH_GRIPPER),),
                "names": [f"target_{n}" for n in TCP_POSE_NAMES_WITH_GRIPPER],
                "info": {
                    "representation": "tcp_pose",
                    "layout": "[x, y, z, rx, ry, rz, gripper]",
                    "unit": "m (xyz), rad axis-angle (rxryrz), bool (gripper)",
                    "frame": "robot_base",
                    "source": "leader/source UR10e RTDE getActualTCPPose",
                    "note": ("Kept for human-readable playback / Isaac "
                             "Sim / UR driver replay. For training "
                             "use ``action.tcp_pose_rot6d`` (continuous)."),
                },
            },
            "action.tcp_pose_rot6d": {
                "dtype": "float32",
                "shape": (len(TCP_POSE_ROT6D_NAMES),),
                "names": [f"target_{n}" for n in TCP_POSE_ROT6D_NAMES],
                "info": {
                    "representation": "tcp_pose_rot6d",
                    "layout": ("[x, y, z, R00, R10, R20, R01, R11, R21,"
                               " gripper]"),
                    "unit": ("m (xyz), unitless (rot6d, first two cols "
                             "of rotation matrix column-major), "
                             "bool (gripper)"),
                    "frame": "robot_base",
                    "source": ("leader/source UR10e RTDE getActualTCPPose"
                               " rotation vector converted via Rodrigues"),
                    "note": ("Continuous rotation representation "
                             "(Zhou et al. 2019). Recommended for "
                             "GR00T / pi0-EE / OpenVLA TCP-action heads."),
                },
            },
            "action.gripper": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["target_is_closed"],
            },
            "action.gripper.position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["target_position"],
                "info": {
                    "representation": "gripper_analog",
                    "unit": "[0,1] Robotiq encoder feedback (pos/255)",
                    "source": ("leader/source Robotiq encoder feedback. "
                               "Standalone column — NOT mirrored into "
                               "``action[6]`` (which holds the discrete "
                               "closed bit)."),
                    "note": ("Physics-affected jaw position on the leader; "
                             "caps around 0.9 even when fully clamped. "
                             "Companion to ``action.gripper``."),
                },
            },
            f"observation.images.{self.camera1_role}": {
                "dtype": "video" if self.use_videos else "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
                "info": {
                    "role": self.camera1_role,
                    "mount": "scene" if self.camera1_role == "scene" else (
                        "wrist" if self.camera1_role == "wrist" else "unknown"),
                    "source_topic": self.color_topic,
                },
            },
        }
        if self.record_depth:
            # LeRobot's image writer requires 3-channel uint8. Pack uint16 mm
            # depth losslessly into (R=high byte, G=low byte, B=0).
            feats[f"observation.images.{self.camera1_role}_depth"] = {
                "dtype": "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
                "info": {
                    "role": self.camera1_role,
                    "encoding": "uint16-mm-packed (R=hi, G=lo, B=0)",
                    "source_topic": self.depth_topic,
                },
            }
        if self.record_camera2:
            feats[f"observation.images.{self.camera2_role}"] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
                "info": {
                    "role": self.camera2_role,
                    "mount": "wrist" if self.camera2_role == "wrist" else (
                        "scene" if self.camera2_role == "scene" else "unknown"),
                    "source_topic": self.color_topic2,
                },
            }
            if self.record_depth:
                feats[f"observation.images.{self.camera2_role}_depth"] = {
                    "dtype": "image",
                    "shape": (self.image_height, self.image_width, 3),
                    "names": ["height", "width", "channels"],
                    "info": {
                        "role": self.camera2_role,
                        "encoding": "uint16-mm-packed (R=hi, G=lo, B=0)",
                        "source_topic": self.depth_topic2,
                    },
                }
        return feats

    def _open_or_create_dataset(self) -> LeRobotDataset:
        # Strategy: always create a fresh dataset per recorder session in a
        # timestamped subdirectory. This avoids two pitfalls of the installed
        # lerobot version:
        #   1. ``LeRobotDataset.create`` calls ``root.mkdir(exist_ok=False)`` and
        #      crashes if any earlier run left the directory behind.
        #   2. Re-opening an existing dataset triggers a HuggingFace Hub lookup
        #      of the (fictitious) ``local/ur10_mirror`` repo, which 401s when
        #      the user is not authenticated.
        # Multiple DI0 episodes within one session still accumulate inside the
        # same dataset; only restarting the launcher creates a new one.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Folder convention: ``session_{robot_type}_{task_slug}_{ts}`` so the
        # directory name shows at a glance which robot the data came from and
        # what the operator labelled it. ``robot_type`` is whatever the launcher
        # (or GUI) sets; ``task_slug`` is the current task description sanitised
        # to filesystem-safe chars.
        robot_slug = self._slugify(self.robot_type or "unknown")
        task_slug = self._slugify(self.task or "episode")
        session_root = self.root / f"session_{robot_slug}_{task_slug}_{ts}"
        session_root.parent.mkdir(parents=True, exist_ok=True)

        # ``LeRobotDataset.create`` itself does the mkdir of ``session_root``
        # with exist_ok=False, so make sure it does not exist yet.
        if session_root.exists():
            shutil.rmtree(session_root)

        self.get_logger().info(f"Creating new dataset @ {session_root}")
        # Codec selection. LeRobot's default ``libsvtav1`` (AV1) is not built
        # into the system ffmpeg on Ubuntu 22.04 / JetPack 6.2 and is therefore
        # unusable here. ``h264_nvenc`` is *listed* by pyav on Jetson Tegra but
        # the runtime init deadlocks because Tegra's encoder is exposed via
        # NVMEDIA/V4L2, not the desktop CUDA NVENC SDK — using it freezes the
        # streaming encoder thread on the first frame, with no error and no
        # progress. Default to ``h264`` (libx264, software): with streaming
        # encoding on, ~640×480×15fps for two cameras is a small percentage of
        # one Orin core and runs in parallel with capture, so save_episode() is
        # still ~near-instant. Override via ``LEROBOT_VCODEC``. Valid values:
        # ``h264``, ``hevc``, ``libsvtav1``, ``auto`` (avoid on Tegra), or
        # hardware codecs like ``h264_nvenc`` (broken on Tegra) /
        # ``h264_v4l2m2m`` (Tegra-native, untested here).
        vcodec = os.environ.get("LEROBOT_VCODEC", "h264")
        if vcodec in ("auto", "h264_nvenc", "hevc_nvenc"):
            self.get_logger().warn(
                f"LEROBOT_VCODEC={vcodec!r} requested — note: on Jetson "
                f"Tegra (Orin/Xavier) NVENC via the CUDA SDK is *not* "
                f"available even though pyav lists it; the encoder will "
                f"hang on first frame. Use ``h264`` instead.")
        # Streaming encoding pushes each frame directly into the encoder as it
        # arrives instead of buffering PNGs on disk and ffmpeg-encoding them at
        # save_episode() time. With this on, save_episode() becomes near-instant
        # (just a final flush + the parquet write); without it, every Stop blocks
        # for the full ffmpeg pass over the per-frame PNGs. Disable with
        # ``LEROBOT_STREAMING_ENCODING=0`` if you ever need to debug.
        streaming_encoding = (
            self.use_videos
            and os.environ.get("LEROBOT_STREAMING_ENCODING", "1") != "0")
        # Image writer is only needed for ``dtype="image"`` features (depth
        # PNGs). With streaming encoding on and no depth, we can skip the writer
        # pool entirely; that also removes the ``_wait_image_writer()`` stall
        # inside save_episode().
        needs_image_writer = (not streaming_encoding) or self.record_depth
        image_writer_processes = 1 if needs_image_writer else 0
        image_writer_threads = 4 if needs_image_writer else 0
        ds = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=session_root,
            robot_type=self.robot_type,
            features=self._features(),
            use_videos=self.use_videos,
            vcodec=vcodec,
            streaming_encoding=streaming_encoding,
            # Flush meta/episodes/*.parquet after every save_episode so a Ctrl+C
            # exit can never lose unflushed episode metadata.
            metadata_buffer_size=1,
            # Async PNG writers — without these, ``add_frame`` writes every
            # camera PNG synchronously on the timer thread which starves the 15
            # Hz timer (we measured ~6 fps actual). One background process w/ a
            # few threads is plenty for two 640×480 RGB streams on the Orin.
            # Skipped entirely when streaming encoding is on and depth is off.
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )
        self.get_logger().info(
            f"Dataset opened — vcodec={vcodec!r} "
            f"streaming_encoding={streaming_encoding} "
            f'image_writer={"on" if needs_image_writer else "off"}')
        # ``modality.json`` is read at training time by GR00T to slice the flat
        # ``action`` / ``state`` vectors into named groups. We always emit it so
        # GR00T users have a working default; ACT / SmolVLA / pi0 ignore the file
        # entirely.
        self._write_modality_json(session_root)
        return ds

    def _write_modality_json(self, session_root: Path) -> None:
        """Emit a GR00T-compatible ``meta/modality.json``.

        Read at *training* time by GR00T's LeRobot data loader to slice the flat
        ``observation.state`` / ``action`` vectors into named modality groups by
        index range. Layout::

            observation.state = [j1..j6, gripper]    (dest joints)
            action            = [j1..j6, gripper]    (source joints)
            observation.tcp_pose = [x,y,z,rx,ry,rz, gripper] (dest TCP)
            action.tcp_pose      = [x,y,z,rx,ry,rz, gripper] (source TCP)

        Has no effect on data collection; consumed only when training.
        """
        modality = {
            "state": {
                "single_arm": {"start": 0, "end": 6},
                "gripper": {"start": 6, "end": 7},
            },
            "action": {
                "single_arm": {"start": 0, "end": 6},
                "gripper": {"start": 6, "end": 7},
            },
            # Alternate task-space slices. To train GR00T on TCP-action instead
            # of joint-action, copy the ``action.tcp_pose`` block into ``action``
            # and set ``original_key="action.tcp_pose"`` in your GR00T config.
            "observation.tcp_pose": {
                "cartesian_position": {"start": 0, "end": 3},
                "axis_angle": {"start": 3, "end": 6},
                "gripper": {"start": 6, "end": 7},
                "original_key": "observation.tcp_pose",
            },
            "action.tcp_pose": {
                "cartesian_position": {"start": 0, "end": 3},
                "axis_angle": {"start": 3, "end": 6},
                "gripper": {"start": 6, "end": 7},
                "original_key": "action.tcp_pose",
            },
            # Continuous-rotation slicers (preferred for training).
            "observation.tcp_pose_rot6d": {
                "cartesian_position": {"start": 0, "end": 3},
                "rotation_6d": {"start": 3, "end": 9},
                "gripper": {"start": 9, "end": 10},
                "original_key": "observation.tcp_pose_rot6d",
            },
            "action.tcp_pose_rot6d": {
                "cartesian_position": {"start": 0, "end": 3},
                "rotation_6d": {"start": 3, "end": 9},
                "gripper": {"start": 9, "end": 10},
                "original_key": "action.tcp_pose_rot6d",
            },
            "video": {
                self.camera1_role: {
                    "original_key": f"observation.images.{self.camera1_role}",
                },
            },
            "annotation": {
                "human.task_description": {},
            },
        }
        if self.record_camera2:
            modality["video"][self.camera2_role] = {
                "original_key": f"observation.images.{self.camera2_role}",
            }
        meta_dir = session_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        path = meta_dir / "modality.json"
        with open(path, "w") as f:
            json.dump(modality, f, indent=2)
        self.get_logger().info(f"Wrote GR00T modality config: {path}")

    # ── Subscriber callbacks ─────────────────────────────────

    @staticmethod
    def _slugify(text: str, max_len: int = 40) -> str:
        """Filesystem-safe slug: keep [A-Za-z0-9_-], swap others for ``_``.

        Used to fold ``robot_type`` and ``task`` into the session folder name
        (``session_{robot}_{task}_{ts}``). Trailing underscores are trimmed and
        the result is capped at ``max_len`` chars so deep task descriptions don't
        blow past the filesystem limit.
        """
        if not text:
            return "unknown"
        out = []
        for ch in str(text):
            if ch.isalnum() or ch in ("_", "-"):
                out.append(ch)
            elif ch.isspace() or ch in ("/", "\\", ".", ",", ":", ";",
                                        "(", ")", "[", "]"):
                out.append("_")
            # Drop anything else (quotes, emoji, control chars).
        slug = "".join(out).strip("_-")
        # Collapse runs of underscores so "pick  up" doesn't become "pick__up".
        while "__" in slug:
            slug = slug.replace("__", "_")
        return (slug[:max_len].rstrip("_-")) or "unknown"

    @staticmethod
    def _reorder_joints(msg: JointState) -> np.ndarray | None:
        """Return joints in canonical UR order, or None if mapping fails."""
        if not msg.name or not msg.position:
            return None
        try:
            idx = [msg.name.index(n) for n in JOINT_NAMES]
        except ValueError:
            # If names not present, assume already in order (length must match).
            if len(msg.position) >= 6:
                return np.asarray(msg.position[:6], dtype=np.float32)
            return None
        return np.asarray([msg.position[i] for i in idx], dtype=np.float32)

    def _obs_joints_cb(self, msg: JointState) -> None:
        v = self._reorder_joints(msg)
        if v is not None:
            with self._lock:
                self._obs_joints = v

    def _act_joints_cb(self, msg: JointState) -> None:
        v = self._reorder_joints(msg)
        if v is not None:
            with self._lock:
                self._act_joints = v

    def _obs_gripper_pos_cb(self, msg: Float64) -> None:
        with self._lock:
            self._obs_gripper_pos = float(msg.data)

    def _obs_gripper_closed_cb(self, msg: Bool) -> None:
        with self._lock:
            self._obs_gripper_closed = bool(msg.data)

    def _act_gripper_pos_cb(self, msg: Float64) -> None:
        with self._lock:
            self._act_gripper_pos = float(msg.data)

    def _act_gripper_closed_cb(self, msg: Bool) -> None:
        with self._lock:
            self._act_gripper_closed = bool(msg.data)

    def _color_cb(self, msg: Image) -> None:
        img = _decode_color_msg(msg)
        if img is None:
            self.get_logger().warn(
                f"color decode failed: unsupported encoding {msg.encoding!r}")
            return
        with self._lock:
            self._latest_color = img

    def _depth_cb(self, msg: Image) -> None:
        img = _decode_depth_msg(msg)
        if img is None:
            self.get_logger().warn(
                f"depth decode failed: unsupported encoding {msg.encoding!r}")
            return
        with self._lock:
            self._latest_depth = img

    def _color2_cb(self, msg: Image) -> None:
        img = _decode_color_msg(msg)
        if img is None:
            self.get_logger().warn(
                f"color2 decode failed: unsupported encoding {msg.encoding!r}")
            return
        with self._lock:
            self._latest_color2 = img

    def _depth2_cb(self, msg: Image) -> None:
        img = _decode_depth_msg(msg)
        if img is None:
            self.get_logger().warn(
                f"depth2 decode failed: unsupported encoding {msg.encoding!r}")
            return
        with self._lock:
            self._latest_depth2 = img

    def _obs_tcp_pose_cb(self, msg: Float64MultiArray) -> None:
        if not msg.data or len(msg.data) < 6:
            return
        with self._lock:
            self._obs_tcp_pose = np.asarray(msg.data[:6], dtype=np.float32)

    def _act_tcp_pose_cb(self, msg: Float64MultiArray) -> None:
        if not msg.data or len(msg.data) < 6:
            return
        with self._lock:
            self._act_tcp_pose = np.asarray(msg.data[:6], dtype=np.float32)

    def _publish_saving(self, value: bool) -> None:
        """Publish the latched ``/recorder/saving`` status.

        Safe to call before ``_saving_pub`` exists (during early init).
        """
        pub = getattr(self, "_saving_pub", None)
        if pub is None:
            return
        msg = Bool()
        msg.data = bool(value)
        pub.publish(msg)

    # ── Episode boundary handling ────────────────────────────

    def _active_cb(self, msg: Bool) -> None:
        # Diagnostic: log every message received, including no-ops, so we can
        # correlate unexpected episode splits with publisher activity.
        self.get_logger().info(
            f"/recorder/active rx data={msg.data} "
            f"is_recording={self.is_recording} "
            f"is_saving={self.is_saving} "
            f"frames_in_ep={self.frames_in_episode}")
        # While the previous episode is still being flushed (ffmpeg + HuggingFace
        # map can take 10-20 s on Jetson), stash the most recent intent and
        # replay it once the save finishes. That keeps the user's clicks from
        # being silently dropped — they can press Start as soon as they want and
        # the next episode begins the instant the recorder is ready.
        if self.is_saving:
            self._pending_active = bool(msg.data)
            self.get_logger().info(
                f"/recorder/active queued during save → pending="
                f"{self._pending_active}")
            return
        if msg.data and not self.is_recording:
            self._start_episode()
        elif not msg.data and self.is_recording:
            self._stop_episode()

    def _start_episode(self) -> None:
        self.frames_in_episode = 0
        self._episode_corrupt = False
        self.is_recording = True
        self.get_logger().info(
            f"🔴 EPISODE {self.episode_count + 1} START — "
            f'task="{self.task}"')

    def _new_session_cb(self, msg: Bool) -> None:
        """Rotate to a fresh dataset directory on a True pulse.

        Refused while a recording is in progress or an episode is still being
        saved — the user should stop and let the save finish first.
        """
        if not msg.data:
            return
        if self.is_recording or self.is_saving:
            self.get_logger().warn(
                "/recorder/new_session ignored — recording or save in "
                "progress; stop the current episode first.")
            return
        # Finalize the OUTGOING dataset before opening the new one, so its
        # parquet footers (data + meta/episodes) and info.json are written to
        # disk. ``metadata_buffer_size=1`` flushes per save_episode but the final
        # ``finalize()`` is what writes the parquet ``PAR1`` magic — without it
        # the rotated-away session is unreadable (pyarrow: "Parquet magic bytes
        # not found in footer"). This used to only happen on Ctrl+C; now rotation
        # produces the same clean result.
        try:
            self.dataset.finalize()
            self.get_logger().info("Previous session finalized cleanly")
        except Exception as e:
            self.get_logger().error(
                f"Previous dataset.finalize failed (rotating anyway): {e}")
        try:
            self.get_logger().info("Rotating to a new session directory...")
            new_ds = self._open_or_create_dataset()
        except Exception as e:
            self.get_logger().error(f"New session failed: {e}")
            return
        self.dataset = new_ds
        self.episode_count = 0
        self.frames_in_episode = 0
        self.get_logger().info("New session active.")

    def _task_description_cb(self, msg: String) -> None:
        """Update the per-frame task string (language instruction)."""
        new_task = (msg.data or "").strip()
        if not new_task:
            return
        if new_task == self.task:
            return
        if self.is_recording:
            self.get_logger().warn(
                f"Task update ignored while recording — current episode "
                f'will keep task="{self.task}". Stop first to change.')
            return
        self.task = new_task
        self.get_logger().info(f'Task description updated → "{self.task}"')

    def _robot_type_cb(self, msg: String) -> None:
        """Update the robot-type label used by the next dataset rotation."""
        new_rt = (msg.data or "").strip()
        if not new_rt or new_rt == (self.robot_type or ""):
            return
        if self.is_recording:
            self.get_logger().warn(
                f"Robot-type update ignored while recording — current "
                f'session keeps robot_type="{self.robot_type}". Stop '
                f"first to change.")
            return
        self.robot_type = new_rt
        self.get_logger().info(
            f'Robot type updated → "{self.robot_type}" '
            f"(applies to next session rotation).")

    def _stop_episode(self) -> None:
        self.is_recording = False
        if self.frames_in_episode == 0:
            self.get_logger().warn("Episode ended with 0 frames — discarded")
            # No frames added → episode_buffer may be None; nothing to clean.
            return
        # Discard corrupt episodes: ``add_frame`` failed mid-loop at some point
        # and the buffer column lengths are now inconsistent. Attempting
        # ``save_episode`` would either raise (cascading the KeyError 'size'
        # corruption to every future episode) or write a mis-aligned
        # parquet/video pair.
        if self._episode_corrupt:
            self.get_logger().error(
                f"Episode marked corrupt after {self.frames_in_episode} "
                f"frames — discarding to prevent dataset wedge")
            try:
                with self._dataset_lock:
                    self.dataset.clear_episode_buffer(
                        delete_images=len(self.dataset.meta.image_keys) > 0)
            except Exception as e:
                self.get_logger().error(f"clear_episode_buffer failed: {e}")
            self._episode_corrupt = False
            return
        if self.frames_in_episode < self.min_episode_frames:
            self.get_logger().warn(
                f"Episode ended with only {self.frames_in_episode} frames "
                f"(< min_episode_frames={self.min_episode_frames}) — "
                f"discarding (likely stray toggle)")
            try:
                with self._dataset_lock:
                    self.dataset.clear_episode_buffer(
                        delete_images=len(self.dataset.meta.image_keys) > 0)
            except Exception as e:
                self.get_logger().error(f"clear_episode_buffer failed: {e}")
            return
        self.is_saving = True
        self._publish_saving(True)
        try:
            with self._dataset_lock:
                self.dataset.save_episode()
            self.episode_count += 1
            self.get_logger().info(
                f"⏹  EPISODE {self.episode_count} SAVED — "
                f"{self.frames_in_episode} frames")
        except Exception as e:
            self.get_logger().error(f"save_episode failed: {e}")
            # ``save_episode`` may have already popped ``size`` from the buffer
            # before raising. Force a fresh buffer so the next episode does not
            # inherit the corruption.
            try:
                with self._dataset_lock:
                    self.dataset.clear_episode_buffer(
                        delete_images=len(self.dataset.meta.image_keys) > 0)
            except Exception as ce:
                self.get_logger().error(
                    f"post-failure clear_episode_buffer failed: {ce}")
        finally:
            self.is_saving = False
            self._publish_saving(False)
            # Replay any /recorder/active edge that arrived during the save. Only
            # acts when the queued state differs from the current is_recording
            # flag so we don't spuriously start a new episode after a clean Stop
            # → (idle) sequence.
            pending, self._pending_active = self._pending_active, None
            if pending is not None:
                if pending and not self.is_recording:
                    self.get_logger().info(
                        "Replaying queued Start → beginning next episode")
                    self._start_episode()
                elif not pending and self.is_recording:
                    self.get_logger().info(
                        "Replaying queued Stop → ending episode")
                    self._stop_episode()

    # ── Frame sampling ───────────────────────────────────────

    def _frame_tick(self) -> None:
        if not self.is_recording or self.is_saving or self._episode_corrupt:
            # Skip ticks during save (would race with save_episode buffer
            # mutation) and after corruption (would pile more bad frames onto a
            # buffer that's getting discarded at stop time).
            return
        with self._lock:
            obs_j = self._obs_joints
            obs_tcp = self._obs_tcp_pose
            obs_grip = self._obs_gripper_pos
            obs_grip_closed = self._obs_gripper_closed
            act_j = self._act_joints
            act_tcp = self._act_tcp_pose
            act_grip = self._act_gripper_pos
            act_grip_closed = self._act_gripper_closed
            color = self._latest_color
            depth = self._latest_depth
            color2 = self._latest_color2
            depth2 = self._latest_depth2

        # All required signals must have been received at least once. When any
        # are missing we drop the frame and (rate-limited) tell the user exactly
        # which signal is blocking, so a misconfigured camera doesn't silently
        # produce a zero-frame dataset.
        missing = []
        if obs_j is None:
            missing.append("obs_joints(/destination/joint_states)")
        if obs_tcp is None:
            missing.append("obs_tcp_pose(/destination/tcp_pose)")
        if act_j is None:
            missing.append("act_joints(/mirror/joint_states)")
        if act_tcp is None:
            missing.append("act_tcp_pose(/mirror/tcp_pose)")
        if color is None:
            missing.append("color")
        if self.record_depth and depth is None:
            missing.append("depth")
        if self.record_camera2 and color2 is None:
            missing.append("color2")
        if self.record_camera2 and self.record_depth and depth2 is None:
            missing.append("depth2")
        if missing:
            now = time.monotonic()
            if now - getattr(self, "_last_drop_log", 0.0) >= 2.0:
                self._last_drop_log = now
                self.get_logger().warn(
                    f"_frame_tick dropping frame — missing: {missing}. "
                    "If this persists, the stream is not publishing; "
                    "either restart the camera with the right launch "
                    "flags or disable the corresponding recorder option.")
            return

        # ── observation.state[6]: destination gripper (BIT) ──
        # Gripper slot in joint and TCP vectors is the discrete closed bit (0.0 /
        # 1.0). Robotiq encoder analog feedback is not recorded in joint/TCP
        # vectors (it never reaches 1.0 even when fully clamped — see
        # ``observation.gripper.is_closed`` / ``action.gripper`` for the
        # canonical discrete signal).
        obs_grip_bit = np.float32(1.0 if obs_grip_closed else 0.0)
        act_grip_bit = np.float32(1.0 if act_grip_closed else 0.0)

        obs_state = np.concatenate([obs_j, [obs_grip_bit]]).astype(np.float32)
        obs_tcp_full = np.concatenate(
            [obs_tcp, [obs_grip_bit]]).astype(np.float32)
        # rot6d alternates: continuous, no wrap. xyz + 6 floats + gripper bit.
        obs_tcp_rot6d_full = np.concatenate(
            [obs_tcp[:3], _axis_angle_to_rot6d(obs_tcp[3:6]),
             [obs_grip_bit]]).astype(np.float32)

        # ── canonical ``action`` column ──
        # Joints + gripper closed bit (0.0 / 1.0).
        action = np.concatenate(
            [act_j, [act_grip_bit]]
        ).astype(np.float32)

        # Extension action vectors:
        #   ``action.joints`` is a strict duplicate of ``action`` (bit gripper).
        #   ``action.tcp_pose`` and ``action.tcp_pose_rot6d`` also end in the
        #   gripper bit — the schema is uniform.
        act_joints_full = action
        act_tcp_full = np.concatenate(
            [act_tcp, [act_grip_bit]]).astype(np.float32)
        act_tcp_rot6d_full = np.concatenate(
            [act_tcp[:3], _axis_angle_to_rot6d(act_tcp[3:6]),
             [act_grip_bit]]).astype(np.float32)

        # Resize images to declared shape if necessary.
        color_f = self._fit_image(color, channels=3)
        frame = {
            # Observation (DESTINATION robot)
            "observation.state": obs_state,
            "observation.tcp_pose": obs_tcp_full,
            "observation.tcp_pose_rot6d": obs_tcp_rot6d_full,
            "observation.gripper.is_closed": np.array(
                [obs_grip_closed], dtype=bool),
            "observation.gripper.position": np.array(
                [obs_grip], dtype=np.float32),
            # Canonical action
            "action": action,
            # Extension action columns (SOURCE robot, always recorded)
            "action.joints": act_joints_full,
            "action.tcp_pose": act_tcp_full,
            "action.tcp_pose_rot6d": act_tcp_rot6d_full,
            "action.gripper": np.array([act_grip_closed], dtype=bool),
            "action.gripper.position": np.array(
                [act_grip], dtype=np.float32),
            # Images + language
            f"observation.images.{self.camera1_role}": color_f,
            "task": self.task,
        }
        if self.record_depth:
            depth_packed = self._pack_depth_u16(depth)
            frame[f"observation.images.{self.camera1_role}_depth"] = depth_packed
        if self.record_camera2:
            frame[f"observation.images.{self.camera2_role}"] = self._fit_image(color2, channels=3)
            if self.record_depth:
                frame[f"observation.images.{self.camera2_role}_depth"] = self._pack_depth_u16(depth2)

        try:
            with self._dataset_lock:
                self.dataset.add_frame(frame)
            self.frames_in_episode += 1
        except Exception as e:
            self.get_logger().error(
                f"add_frame failed: {e} — marking episode corrupt, "
                f"will discard on stop")
            self._episode_corrupt = True

    def _pack_depth_u16(self, depth: np.ndarray) -> np.ndarray:
        """uint16 depth (mm) -> (H,W,3) uint8 with R=high byte, G=low byte."""
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16, copy=False)
        h, w = self.image_height, self.image_width
        if depth.shape != (h, w):
            try:
                import cv2
                depth = cv2.resize(depth, (w, h),
                                   interpolation=cv2.INTER_NEAREST)
            except ImportError:
                depth = depth[:h, :w]
        hi = (depth >> 8).astype(np.uint8)
        lo = (depth & 0xFF).astype(np.uint8)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[..., 0] = hi
        out[..., 1] = lo
        return out

    def _fit_image(self, img: np.ndarray, channels: int,
                   dtype: Any = np.uint8) -> np.ndarray:
        """Center-crop / pad to (image_height, image_width, channels)."""
        h, w = self.image_height, self.image_width
        if img.shape[0] != h or img.shape[1] != w:
            try:
                import cv2
                img = cv2.resize(img, (w, h),
                                 interpolation=cv2.INTER_NEAREST
                                 if dtype == np.uint16 else cv2.INTER_AREA)
            except ImportError:
                img = img[:h, :w]
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[2] != channels:
            img = img[..., :channels]
        return img.astype(dtype, copy=False)

    # ── Periodic status ──────────────────────────────────────

    def _log_tick(self) -> None:
        status = "RECORDING" if self.is_recording else "IDLE"
        self.get_logger().info(
            f"[{status}] episodes={self.episode_count} "
            f"frames_this_ep={self.frames_in_episode}")

    # ── Cleanup ──────────────────────────────────────────────

    def destroy_node(self) -> None:
        if self.is_recording:
            self.get_logger().info("Shutdown — flushing active episode")
            self._stop_episode()
        # Explicitly finalize the dataset so meta/episodes parquet, stats, and
        # info.json are all flushed to disk before exit. Without this we rely on
        # __del__ which Python may skip on Ctrl+C.
        try:
            self.dataset.finalize()
            self.get_logger().info("Dataset finalized cleanly")
        except Exception as e:
            self.get_logger().error(f"dataset.finalize failed: {e}")
        super().destroy_node()


def _acquire_singleton_lock() -> Any:
    """Prevent two recorder processes from racing on the same dataset.

    Two concurrent ``lerobot_recorder_node.py`` processes would both subscribe
    to ``/recorder/active`` and each open a private dataset folder; episodes
    would be silently split between the two, with one folder ending up empty or
    partial. We guard against that with a PID-file flock — the second invocation
    refuses to start.
    """
    import atexit
    import fcntl
    lock_path = "/tmp/ur10e_lerobot_recorder.pid"
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Read the holder PID for a useful error message.
        try:
            with open(lock_path) as r:
                holder = r.read().strip() or "?"
        except Exception:
            holder = "?"
        _LOGGER.error(
            "another lerobot_recorder_node.py is already running (pid=%s). "
            "Refusing to start a second one — it would race on the dataset "
            "folder. Kill it first:\n  pkill -9 -f lerobot_recorder_node.py",
            holder)
        sys.exit(1)
    fh.write(f"{os.getpid()}\n")
    fh.flush()

    def _release() -> None:
        with contextlib.suppress(Exception):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
            os.unlink(lock_path)

    atexit.register(_release)
    # Returned so the caller keeps a strong reference (and the lock).
    return fh


def main(args: list[str] | None = None) -> None:
    # Acquire the single-instance lock BEFORE rclpy.init so the second process
    # exits cleanly without spamming ROS topics.
    _LOCK_FH = _acquire_singleton_lock()  # noqa: F841 - hold the lock
    rclpy.init(args=args)
    node = LeRobotRecorderNode()
    # Multi-threaded executor so image, joint, gripper, tcp and timer callbacks
    # don't serialize on a single thread. With single-threaded spin the 15 Hz
    # timer was getting starved by image-decode callbacks and only producing ~6
    # fps of recorded frames.
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().info("Shutting down LeRobot Recorder Node")
    finally:
        executor.shutdown()
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
