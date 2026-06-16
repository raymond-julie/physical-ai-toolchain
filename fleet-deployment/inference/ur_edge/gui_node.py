#!/usr/bin/env python3
"""GUI node for the UR edge runtime.

Flask + SocketIO web GUI served as a ROS 2 node. Provides:

* Real-time robot state via WebSocket
* Live camera preview via MJPEG stream
* GUI-triggered start/stop (publishes /gui/toggle)
* Recording list, playback, deletion
* Runtime settings management

Runs on http://0.0.0.0:8080

Requirements::

    pip install flask flask-socketio
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

from model_runner import ModelRunner
from replay_runner import ReplayRunner

_LOGGER = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "templates")
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, "recordings")
RECORDINGS_RAW_DIR = os.path.join(SCRIPT_DIR, "recordings_raw")
# LeRobot-format datasets — source for episode replay.
RECORDINGS_LEROBOT_DIR = os.path.join(SCRIPT_DIR, "recordings_lerobot")

# Where all model families + checkpoints live. The combo box enumerates
# ``<MODEL_BASE_DIR>/<family>/<version>`` entries; each is auto-classified as
# either a SmolVLA checkpoint (nested ``pretrained_model/``) or an NVIDIA
# Isaac-GR00T checkpoint (``config.json`` with architecture ``GR00T_*``). See
# ``model_runner.list_versions`` for details.
MODEL_BASE_DIR = os.path.join(SCRIPT_DIR, "ai_models")


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["SECRET_KEY"] = os.environ.get("GUI_SECRET_KEY", "ur-recorder-gui")
# Re-render templates on every request so HTML edits show up without restarting
# the GUI. Also disable static-asset caching so app.js / CSS changes are picked
# up by a normal browser refresh.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ── Global state (written by ROS callbacks, read by Flask routes) ────────────
class GuiState:
    """Thread-safe state container shared between ROS 2 and Flask."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

        # Source robot
        self.source_joints = [0.0] * 6
        self.source_gripper_pos = 0.0
        self.source_gripper_closed = False
        self.source_di0 = False
        self.source_di1 = False
        self.source_connected = False
        self.source_last_msg_time = 0.0

        # Source TCP pose [x,y,z,rx,ry,rz] (metres / radians, base frame).
        self.source_tcp = [0.0] * 6
        self.source_tcp_valid = False
        # XY positions captured each time the source gripper rising-edges to
        # closed. Each entry is {x,y,z,t} where t is unix timestamp.
        self.grip_points: list[dict] = []
        # Track previous gripper-closed state for rising-edge detection.
        self._grip_closed_prev = False
        # Hard cap so the list cannot grow without bound.
        self.grip_points_max = 500

        # Destination robot
        self.dest_joints = [0.0] * 6
        self.dest_gripper_pos = 0.0
        self.dest_connected = False
        self.dest_last_msg_time = 0.0
        # Destination TCP pose [x,y,z,rx,ry,rz] (metres + axis-angle rad) in
        # URScript base-frame convention, populated from ``/destination/tcp_pose``.
        # Needed by Cartesian / TCP-EE policies (e.g. the GR00T
        # groot-ur10-tcp-ee-* checkpoint) so the model sees the follower's
        # current pose as state.eef.
        self.dest_tcp = [0.0] * 6
        self.dest_tcp_valid = False

        # State machine
        self.state = "WAITING"
        self.recorder_active = False
        # True while save_episode() is flushing parquet + mp4 in the recorder
        # node. Updated via the latched ``/recorder/saving`` topic. The Start
        # button is disabled in the GUI while this is True so the user knows
        # their click is queued.
        self.recorder_saving = False
        self.motion_confirmed = False

        # Camera 1
        self.color_frame = None       # latest JPEG bytes
        self.depth_frame = None       # latest JPEG bytes (colorized)
        self.color_bgr = None         # latest BGR ndarray (for inference)
        self.camera_connected = False
        self.camera_last_msg_time = 0.0

        # Camera 2
        self.color_frame2 = None
        self.depth_frame2 = None
        self.color_bgr2 = None        # latest BGR ndarray (for inference)
        self.camera2_connected = False
        self.camera2_last_msg_time = 0.0

        # Recording
        self.current_bag_name = None
        self.recording_start_time = None
        self.recording_count = 0
        # Per-episode task / language instruction. Defaults to whatever is
        # persisted in gui_settings.json (loaded in main()).
        self.task_description = "Pick Large White Gear, Place Blue Bin"
        # Robot type label baked into ``info.json`` (``robot_type``) and into the
        # session folder name (``session_{robot}_{task}_{ts}``). Lets the
        # operator label a recording rig from the GUI without editing the
        # launcher. Persisted to gui_settings.json.
        self.robot_type = "UR10_Single"

        # Display preferences. ``show_depth`` controls whether the GUI serves the
        # depth MJPEG streams. Initialised from the launcher's
        # RECORD_DEPTH_DEFAULT env var so it matches the launch flag, but the
        # user can flip it at confirm-motion time.
        self.show_depth = (
            os.environ.get("RECORD_DEPTH_DEFAULT", "false").lower() == "true")

    def snapshot(self) -> dict:
        """Return a JSON-serialisable dict of all state."""
        with self.lock:
            now = time.time()
            src_conn = ((now - self.source_last_msg_time) < 3.0
                        if self.source_last_msg_time > 0 else False)
            dst_conn = ((now - self.dest_last_msg_time) < 3.0
                        if self.dest_last_msg_time > 0 else False)
            cam1_conn = ((now - self.camera_last_msg_time) < 3.0
                         if self.camera_last_msg_time > 0 else False)
            cam2_conn = ((now - self.camera2_last_msg_time) < 3.0
                         if self.camera2_last_msg_time > 0 else False)
            # Recording-ready gate. We refuse to start a recording when any
            # required signal isn't live or when the destination arm isn't
            # positionally tracking the source. Recording during
            # WAITING/ALIGNING/RETURNING captures meaningless action labels
            # because the destination arm is not following the source TCP —
            # those frames poison the dataset.
            blockers = []
            if not src_conn:
                blockers.append("source robot not publishing")
            if not dst_conn:
                blockers.append("destination robot not publishing")
            if not cam1_conn:
                blockers.append("camera 1 (scene) not publishing")
            if not cam2_conn:
                blockers.append("camera 2 (wrist) not publishing")
            if self.state not in ("IDLE", "MIRRORING"):
                blockers.append(f"arms not synced (state={self.state})")
            ready_to_record = len(blockers) == 0
            return {
                "source": {
                    "joints": [round(j, 4) for j in self.source_joints],
                    "gripper_pos": round(self.source_gripper_pos, 3),
                    "gripper_closed": self.source_gripper_closed,
                    "di0": self.source_di0,
                    "di1": self.source_di1,
                    "connected": src_conn,
                },
                "dest": {
                    "joints": [round(j, 4) for j in self.dest_joints],
                    "gripper_pos": round(self.dest_gripper_pos, 3),
                    "connected": dst_conn,
                },
                "state": self.state,
                "motion_confirmed": self.motion_confirmed,
                "recorder_active": self.recorder_active,
                "recorder_saving": self.recorder_saving,
                "tcp": {
                    "valid": self.source_tcp_valid,
                    "x": round(self.source_tcp[0], 4),
                    "y": round(self.source_tcp[1], 4),
                    "z": round(self.source_tcp[2], 4),
                },
                "grip_points": [
                    {"x": round(p["x"], 4),
                     "y": round(p["y"], 4),
                     "z": round(p["z"], 4),
                     "t": p["t"]}
                    for p in self.grip_points
                ],
                "camera_connected": cam1_conn,
                "camera2_connected": cam2_conn,
                "camera_model": os.environ.get("CAM1_MODEL", "") or None,
                "camera2_model": os.environ.get("CAM2_MODEL", "") or None,
                "show_depth": self.show_depth,
                "current_bag": self.current_bag_name,
                "recording_elapsed": round(now - self.recording_start_time, 1)
                if self.recording_start_time else None,
                "recording_count": self.recording_count,
                "ready_to_record": ready_to_record,
                "recording_blockers": blockers,
                "task_description": self.task_description,
                "robot_type": self.robot_type,
                "model": model_runner.status() if model_runner else None,
            }


state = GuiState()


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 Node
# ══════════════════════════════════════════════════════════════════════════════

class GuiRosNode(Node):
    """ROS 2 node that bridges topics to the Flask GUI."""

    JOINT_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]

    def __init__(self) -> None:
        super().__init__("gui_node")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )

        # ── Subscribers ──────────────────────────────────────────

        # Source robot
        self.create_subscription(
            JointState, "/mirror/joint_states", self._src_joint_cb, qos)
        self.create_subscription(
            Float64, "/mirror/gripper/position", self._src_grip_pos_cb, qos)
        self.create_subscription(
            Bool, "/mirror/gripper/is_closed", self._src_grip_closed_cb, qos)
        self.create_subscription(
            Bool, "/mirror/tool_digital_input_0", self._src_di0_cb, qos)
        self.create_subscription(
            Bool, "/mirror/tool_digital_input_1", self._src_di1_cb, qos)
        # Source TCP pose for the gripper-close map.
        self.create_subscription(
            Float64MultiArray, "/mirror/tcp_pose", self._src_tcp_cb, qos)

        # Destination robot
        self.create_subscription(
            JointState, "/joint_states", self._dst_joint_cb, qos)
        self.create_subscription(
            Float64, "/destination/gripper/position", self._dst_grip_pos_cb, qos)
        # Destination TCP pose so Cartesian-policy obs has a current pose.
        self.create_subscription(
            Float64MultiArray, "/destination/tcp_pose",
            self._dst_tcp_cb, qos)

        # Destination state
        self.create_subscription(
            String, "/destination/state", self._dest_state_cb, qos)

        # Recorder
        self.create_subscription(
            Bool, "/recorder/active", self._recorder_active_cb, qos)
        # Latched ``/recorder/saving`` so we know when the recorder is busy
        # flushing the previous episode. Use the matching TRANSIENT_LOCAL QoS or
        # subscriptions will silently drop.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool, "/recorder/saving", self._recorder_saving_cb,
            latched_qos)

        # Camera 1
        self.create_subscription(
            Image, "/camera1/camera1/color/image_raw", self._color_cb, qos)
        self.create_subscription(
            Image, "/camera1/camera1/depth/image_rect_raw", self._depth_cb, qos)

        # Camera 2
        self.create_subscription(
            Image, "/camera2/camera2/color/image_raw", self._color2_cb, qos)
        self.create_subscription(
            Image, "/camera2/camera2/depth/image_rect_raw", self._depth2_cb, qos)

        # ── Publisher (GUI → destination writer) ─────────────────
        self.gui_toggle_pub = self.create_publisher(Bool, "/gui/toggle", qos)
        self.gui_confirm_pub = self.create_publisher(Bool, "/gui/confirm_motion", qos)
        self.gui_use_home_pub = self.create_publisher(Bool, "/gui/use_home", qos)
        # Latched speed-scale publisher so a destination_writer that connects
        # late still picks up the operator's chosen value.
        latch_qos_speed = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.gui_speed_scale_pub = self.create_publisher(
            Float64, "/gui/speed_scale", latch_qos_speed)
        # Latched too: destination_writer that subscribes late still needs the
        # current interpolation scale.
        self.gui_interp_scale_pub = self.create_publisher(
            Float64, "/gui/interp_scale", latch_qos_speed)
        # Operator confirmation that the workspace is safe to home after a
        # protective stop. destination_writer only auto-unlocks once it sees True
        # on this topic.
        self.gui_recovery_ack_pub = self.create_publisher(
            Bool, "/gui/recovery_ack", qos)
        # New-session pulse for the recorder — rotates the dataset to a fresh
        # timestamped folder without restarting the node.
        self.new_session_pub = self.create_publisher(
            Bool, "/recorder/new_session", qos)
        # Per-episode task description (language instruction). The recorder
        # updates self.task and tags each future episode with it.
        latch_qos_local = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.task_description_pub = self.create_publisher(
            String, "/recorder/task_description", latch_qos_local)
        # Robot-type label, same latched-topic pattern so a late-joining recorder
        # still sees the operator's last choice.
        self.robot_type_pub = self.create_publisher(
            String, "/recorder/robot_type", latch_qos_local)
        # ── Publishers used when an AI model drives the destination robot. They
        # republish onto the same topics the source_reader uses, so the existing
        # destination_writer state machine picks them up unchanged. Match
        # source_reader / destination_writer joint QoS exactly: RELIABLE,
        # KEEP_LAST depth=1 — otherwise destination_writer's RELIABLE
        # subscription rejects our samples.
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.model_joint_pub = self.create_publisher(
            JointState, "/mirror/joint_states", joint_qos)
        self.model_gripper_pub = self.create_publisher(
            Float64, "/mirror/gripper/position", qos)
        # Cartesian-replay command path: destination_writer listens on
        # ``/mirror/tcp_pose_cmd`` (distinct from the informational
        # ``/mirror/tcp_pose`` that source_reader publishes during teach-mode)
        # and switches to ``servoL`` while samples are fresh, letting the UR
        # controller's firmware do the IK.
        self.model_tcp_pub = self.create_publisher(
            Float64MultiArray, "/mirror/tcp_pose_cmd", joint_qos)
        # Latched-style flag so source_reader stops echoing the leader robot onto
        # /mirror/* whenever a policy is driving the destination.
        latch_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.policy_active_pub = self.create_publisher(
            Bool, "/policy/active", latch_qos)
        # Initial state: no policy running.
        self._publish_policy_active(False)
        # Cache so we only republish on state change.
        self._last_policy_active = False
        # ── Periodic WebSocket push ─────────────────────────────
        self.create_timer(0.1, self._push_state)  # 10 Hz
        # Watchdog: ensure /policy/active reflects the model_runner's true state.
        # Without this, a model that errored or whose worker exited would leave
        # /policy/active=True forever — which used to silently suppress recording
        # on the destination_writer.
        self.create_timer(0.5, self._policy_active_watchdog)

        self.get_logger().info("GUI Node started — http://0.0.0.0:8080")

    # ── Source callbacks ─────────────────────────────────────────

    def _src_joint_cb(self, msg: JointState) -> None:
        with state.lock:
            state.source_joints = list(msg.position[:6])
            state.source_last_msg_time = time.time()

    def _src_grip_pos_cb(self, msg: Float64) -> None:
        with state.lock:
            state.source_gripper_pos = msg.data

    def _src_grip_closed_cb(self, msg: Bool) -> None:
        with state.lock:
            new_closed = bool(msg.data)
            # Rising edge: gripper just closed -> record TCP xy.
            if (new_closed and not state._grip_closed_prev
                    and state.source_tcp_valid):
                state.grip_points.append({
                    "x": float(state.source_tcp[0]),
                    "y": float(state.source_tcp[1]),
                    "z": float(state.source_tcp[2]),
                    "t": time.time(),
                })
                # Trim to cap.
                if len(state.grip_points) > state.grip_points_max:
                    state.grip_points = state.grip_points[-state.grip_points_max:]
            state._grip_closed_prev = new_closed
            state.source_gripper_closed = new_closed

    def _src_tcp_cb(self, msg: Float64MultiArray) -> None:
        if not msg.data or len(msg.data) < 3:
            return
        with state.lock:
            state.source_tcp = [float(v) for v in list(msg.data)[:6]] + \
                [0.0] * max(0, 6 - len(msg.data))
            state.source_tcp_valid = True

    def _src_di0_cb(self, msg: Bool) -> None:
        with state.lock:
            state.source_di0 = msg.data

    def _src_di1_cb(self, msg: Bool) -> None:
        with state.lock:
            state.source_di1 = msg.data

    # ── Destination callbacks ────────────────────────────────────

    def _dst_joint_cb(self, msg: JointState) -> None:
        with state.lock:
            state.dest_joints = list(msg.position[:6])
            state.dest_last_msg_time = time.time()

    def _dst_grip_pos_cb(self, msg: Float64) -> None:
        with state.lock:
            state.dest_gripper_pos = msg.data

    def _dst_tcp_cb(self, msg: Float64MultiArray) -> None:
        if not msg.data or len(msg.data) < 6:
            return
        with state.lock:
            state.dest_tcp = [float(v) for v in list(msg.data)[:6]]
            state.dest_tcp_valid = True

    # ── Destination state callback ───────────────────────────────

    def _dest_state_cb(self, msg: String) -> None:
        with state.lock:
            state.state = msg.data
            if msg.data != "WAITING":
                state.motion_confirmed = True

    # ── Recorder callback ────────────────────────────────────────

    def _recorder_active_cb(self, msg: Bool) -> None:
        with state.lock:
            was_active = state.recorder_active
            state.recorder_active = msg.data
            if msg.data and not was_active:
                state.recording_start_time = time.time()
                state.recording_count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                state.current_bag_name = f"recording_{ts}"
            elif not msg.data and was_active:
                state.recording_start_time = None
                state.current_bag_name = None

    def _recorder_saving_cb(self, msg: Bool) -> None:
        """Track whether the recorder is currently flushing an episode."""
        with state.lock:
            state.recorder_saving = bool(msg.data)

    # ── Camera callbacks ─────────────────────────────────────────

    @staticmethod
    def _decode_color(msg: Image) -> np.ndarray | None:
        """Return BGR ndarray from a sensor_msgs/Image, or None."""
        if msg.encoding == "rgb8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if msg.encoding == "bgr8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        return None

    @classmethod
    def _encode_color(cls, msg: Image) -> tuple[bytes | None, np.ndarray | None]:
        img = cls._decode_color(msg)
        if img is None:
            return None, None
        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes(), img

    @staticmethod
    def _encode_depth(msg: Image) -> bytes | None:
        if msg.encoding != "16UC1":
            return None
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width)
        depth_norm = cv2.normalize(depth, None, 0, 255,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        _, jpeg = cv2.imencode(".jpg", depth_color,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes()

    def _color_cb(self, msg: Image) -> None:
        with contextlib.suppress(Exception):
            jpeg, bgr = self._encode_color(msg)
            if jpeg is None:
                return
            with state.lock:
                state.color_frame = jpeg
                state.color_bgr = bgr
                state.camera_connected = True
                state.camera_last_msg_time = time.time()

    def _depth_cb(self, msg: Image) -> None:
        with contextlib.suppress(Exception):
            jpeg = self._encode_depth(msg)
            if jpeg is None:
                return
            with state.lock:
                state.depth_frame = jpeg

    def _color2_cb(self, msg: Image) -> None:
        with contextlib.suppress(Exception):
            jpeg, bgr = self._encode_color(msg)
            if jpeg is None:
                return
            with state.lock:
                state.color_frame2 = jpeg
                state.color_bgr2 = bgr
                state.camera2_connected = True
                state.camera2_last_msg_time = time.time()

    def _depth2_cb(self, msg: Image) -> None:
        with contextlib.suppress(Exception):
            jpeg = self._encode_depth(msg)
            if jpeg is None:
                return
            with state.lock:
                state.depth_frame2 = jpeg

    # ── Push state to WebSocket ──────────────────────────────────

    def _push_state(self) -> None:
        with contextlib.suppress(Exception):
            socketio.emit("state_update", state.snapshot(), namespace="/")

    def _policy_active_watchdog(self) -> None:
        """Republish /policy/active=False whenever no driver is active.

        A "driver" is anything that needs destination_writer to be in MIRRORING
        via /policy/active — currently the AI model runner and the episode replay
        runner. The flag goes True if either is running and False once both have
        stopped.

        Without this watchdog the latched True from a previous run could persist
        indefinitely (silently disabling recording on destination_writer) and a
        worker that exits via an exception instead of stop() would never clear
        the flag.
        """
        running = False
        if model_runner is not None:
            try:
                running = running or (
                    model_runner.status().get("state")
                    == ModelRunner.STATE_RUNNING)
            except Exception:
                pass
        if replay_runner is not None:
            try:
                running = running or (
                    replay_runner.status().get("state")
                    in (ReplayRunner.STATE_LOADING,
                        ReplayRunner.STATE_RUNNING))
            except Exception:
                pass
        if running != self._last_policy_active:
            self._publish_policy_active(running)
            self._last_policy_active = running

    # ── GUI toggle (called from Flask route) ─────────────────────

    def publish_toggle(self) -> None:
        """Publish a Bool(True) pulse on /gui/toggle."""
        msg = Bool()
        msg.data = True
        self.gui_toggle_pub.publish(msg)
        self.get_logger().info("GUI toggle published")

    def publish_new_session(self) -> None:
        """Publish a Bool(True) pulse on /recorder/new_session."""
        msg = Bool()
        msg.data = True
        self.new_session_pub.publish(msg)
        self.get_logger().info("New-session pulse published")

    def publish_task_description(self, task: str) -> None:
        """Publish the current task description for the recorder."""
        msg = String()
        msg.data = task or ""
        self.task_description_pub.publish(msg)
        self.get_logger().info(f'Task description published → "{task}"')

    def publish_robot_type(self, robot_type: str) -> None:
        """Publish the current robot-type label for the recorder."""
        msg = String()
        msg.data = robot_type or ""
        self.robot_type_pub.publish(msg)
        self.get_logger().info(f'Robot type published → "{robot_type}"')

    def publish_confirm_motion(self, use_home: bool = True) -> None:
        """Publish use_home choice, then confirm motion."""
        # Send use_home preference first
        home_msg = Bool()
        home_msg.data = use_home
        self.gui_use_home_pub.publish(home_msg)
        self.get_logger().info(f"GUI use_home={use_home} published")

        # Then confirm motion
        msg = Bool()
        msg.data = True
        self.gui_confirm_pub.publish(msg)
        self.get_logger().info("GUI motion confirmation published")
        with state.lock:
            state.motion_confirmed = True

    def publish_speed_scale(self, scale: float) -> None:
        """Publish robot speed scale (0.01..1.0) on /gui/speed_scale."""
        s = max(0.01, min(1.0, float(scale)))
        msg = Float64()
        msg.data = s
        self.gui_speed_scale_pub.publish(msg)
        self.get_logger().info(f"/gui/speed_scale <- {s * 100:.0f}%")

    def publish_interp_scale(self, scale: float) -> None:
        """Publish TCP-interpolation horizon multiplier on /gui/interp_scale.

        0 = OFF (snap), 1.0 = ramp over one measured inter-arrival, >1 lazier,
        <1 snappier.
        """
        s = max(0.0, min(10.0, float(scale)))
        msg = Float64()
        msg.data = s
        self.gui_interp_scale_pub.publish(msg)
        self.get_logger().info(f"/gui/interp_scale <- {s:.2f}")

    def publish_recovery_ack(self) -> None:
        """Pulse True on /gui/recovery_ack so destination_writer can unlock a
        latched protective stop and return to home. Caller must have visually
        confirmed the workspace is clear.
        """
        msg = Bool()
        msg.data = True
        self.gui_recovery_ack_pub.publish(msg)
        self.get_logger().info("GUI recovery acknowledgment published")

    # ── Model-driven publishers ─────────────────────────────

    def _publish_policy_active(self, active: bool) -> None:
        """Tell source_reader whether to suppress its own /mirror/* echoes.

        Published on /policy/active with TRANSIENT_LOCAL durability so a
        source_reader that subscribes after this is sent still sees the latest
        value.
        """
        msg = Bool()
        msg.data = bool(active)
        self.policy_active_pub.publish(msg)
        self._last_policy_active = bool(active)
        self.get_logger().info(f"/policy/active <- {bool(active)}")

    def publish_model_action(self, joint_targets: list[float],
                             gripper_pos: float) -> None:
        """Republish a policy-generated action on the /mirror/* topics so the
        destination_writer state machine drives the robot to it.
        """
        # First publish: log subscriber counts so you can tell whether the
        # destination_writer is actually listening.
        if not getattr(self, "_model_pub_logged", False):
            self._model_pub_logged = True
            try:
                jc = self.model_joint_pub.get_subscription_count()
                gc = self.model_gripper_pub.get_subscription_count()
                self.get_logger().info(
                    f"[model] First publish — /mirror/joint_states subs={jc}, "
                    f"/mirror/gripper/position subs={gc}")
                if jc == 0:
                    self.get_logger().warn(
                        "[model] No subscribers on /mirror/joint_states — "
                        "the destination_writer is not running or not "
                        "subscribed; robot will not move.")
            except Exception as exc:
                self.get_logger().warn(f"[model] sub-count probe failed: {exc}")

        if joint_targets and len(joint_targets) >= 6:
            js = JointState()
            now = self.get_clock().now().to_msg()
            js.header.stamp = now
            js.name = list(self.JOINT_NAMES)
            js.position = [float(j) for j in joint_targets[:6]]
            self.model_joint_pub.publish(js)
        gp = Float64()
        gp.data = float(max(0.0, min(1.0, gripper_pos)))
        self.model_gripper_pub.publish(gp)

        # Periodic confirmation (~1 Hz) that we're publishing.
        now = time.monotonic()
        if now - getattr(self, "_last_model_pub_log", 0.0) >= 1.0:
            self._last_model_pub_log = now
            j_str = (", ".join(f"{j:+.3f}" for j in joint_targets[:6])
                     if joint_targets else "<none>")
            self.get_logger().info(
                f"[model] publish joints=[{j_str}] gripper={gp.data:.3f}")

    def publish_model_tcp_action(self, tcp_pose: list[float],
                                 gripper_pos: float) -> None:
        """Republish a policy-generated Cartesian pose target on
        ``/mirror/tcp_pose_cmd`` so destination_writer drives the follower via
        ``servoL`` (UR controller does the IK).

        ``tcp_pose`` is ``[x, y, z, rx, ry, rz]`` in URScript base-frame
        convention (metres + axis-angle radians). The gripper is still published
        on its own ``/mirror/gripper/position`` topic since that path is
        independent of the IK / FK choice.
        """
        if tcp_pose is not None and len(tcp_pose) >= 6:
            tm = Float64MultiArray()
            tm.data = [float(x) for x in tcp_pose[:6]]
            self.model_tcp_pub.publish(tm)
        gp = Float64()
        gp.data = float(max(0.0, min(1.0, gripper_pos)))
        self.model_gripper_pub.publish(gp)
        # ~1 Hz log so the user can confirm Cartesian publish is live.
        now = time.monotonic()
        if now - getattr(self, "_last_tcp_pub_log", 0.0) >= 1.0:
            self._last_tcp_pub_log = now
            t_str = (", ".join(f"{v:+.3f}" for v in tcp_pose[:6])
                     if tcp_pose else "<none>")
            self.get_logger().info(
                f"[replay-tcp] publish pose=[{t_str}] gripper={gp.data:.3f}")


# ── Global references (set in main) ──────────────────────────────────────
ros_node: GuiRosNode | None = None
model_runner: ModelRunner | None = None
replay_runner: ReplayRunner | None = None


def _model_observation_provider() -> dict | None:
    """Build a snapshot of the current observation for the policy."""
    with state.lock:
        joints = list(state.dest_joints)
        gripper = float(state.dest_gripper_pos)
        tcp = list(state.dest_tcp) if state.dest_tcp_valid else None
        cam1 = state.color_bgr.copy() if state.color_bgr is not None else None
        cam2 = state.color_bgr2.copy() if state.color_bgr2 is not None else None
    if cam1 is None or cam2 is None:
        # Rate-limited diagnostic so the user understands why model is idle.
        now = time.monotonic()
        if now - getattr(_model_observation_provider, "_last_warn", 0.0) >= 2.0:
            _model_observation_provider._last_warn = now
            if ros_node is not None:
                ros_node.get_logger().warn(
                    f"[model] obs unavailable: cam1={cam1 is not None} "
                    f"cam2={cam2 is not None} joints_len={len(joints)} "
                    "— waiting for camera frames")
        return None
    return {
        "joints": joints,
        "gripper": gripper,
        "tcp_pose": tcp,
        "camera1": cam1,
        "camera2": cam2,
        # No third physical camera available. Training data only contained two
        # cameras (``color``, ``color2``); SmolVLA's prepare_images zero-pads the
        # missing third slot internally. Sending nothing here is intentional — do
        # NOT duplicate cam2.
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Flask routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/state")
def api_state() -> Response:
    return jsonify(state.snapshot())


# ── Start / Stop toggle ─────────────────────────────────────────────────────

@app.route("/api/toggle", methods=["POST"])
def api_toggle() -> Response | tuple[Response, int]:
    """Software-trigger mirroring/recording toggle (same as DI0 press)."""
    if not ros_node:
        return jsonify({"ok": False, "msg": "ROS node not ready"}), 503
    # If we're about to START a recording (recorder currently inactive), enforce
    # the readiness gate. Stopping is always allowed so the user can recover from
    # a bad state.
    snap = state.snapshot()
    if not snap.get("recorder_active") and not snap.get("ready_to_record", True):
        reasons = snap.get("recording_blockers", [])
        return jsonify({
            "ok": False,
            "msg": "Cannot start recording: " + "; ".join(reasons),
            "blockers": reasons,
        }), 409
    ros_node.publish_toggle()
    return jsonify({"ok": True, "msg": "Toggle published"})


@app.route("/api/grip_points/clear", methods=["POST"])
def api_grip_points_clear() -> Response:
    """Clear the gripper-close XY plot."""
    with state.lock:
        n = len(state.grip_points)
        state.grip_points = []
    return jsonify({"ok": True, "cleared": n})


@app.route("/api/new_session", methods=["POST"])
def api_new_session() -> Response | tuple[Response, int]:
    """Start a fresh recorder session folder (timestamped dataset dir).

    Refused while a recording is in progress. The recorder node listens on
    ``/recorder/new_session`` and rotates its dataset on the pulse.
    """
    if not ros_node:
        return jsonify({"ok": False, "msg": "ROS node not ready"}), 503
    snap = state.snapshot()
    if snap.get("recorder_active"):
        return jsonify({
            "ok": False,
            "msg": "Stop the current recording before starting a new session.",
        }), 409
    ros_node.publish_new_session()
    # Reset GUI session counter so the badge restarts at 0.
    with state.lock:
        state.recording_count = 0
        state.grip_points = []
    return jsonify({"ok": True, "msg": "New session pulse published"})


@app.route("/api/task_description", methods=["POST"])
def api_set_task() -> Response | tuple[Response, int]:
    """Update the per-episode task / language instruction.

    Refused mid-recording so the running episode keeps a stable tag. The string
    is persisted to gui_settings.json so it survives restarts.
    """
    if not ros_node:
        return jsonify({"ok": False, "msg": "ROS node not ready"}), 503
    data = request.get_json() or {}
    task = str(data.get("task_description", "")).strip()
    if not task:
        return jsonify({"ok": False, "msg": "task_description required"}), 400
    snap = state.snapshot()
    if snap.get("recorder_active"):
        return jsonify({
            "ok": False,
            "msg": "Stop the current recording before changing the task.",
        }), 409
    with state.lock:
        state.task_description = task
    ros_node.publish_task_description(task)
    try:
        settings = _load_settings()
        settings["task_description"] = task
        _save_settings(settings)
    except Exception as exc:
        ros_node.get_logger().warn(f"Could not persist task: {exc}")
    return jsonify({"ok": True, "task_description": task})


@app.route("/api/robot_type", methods=["POST"])
def api_set_robot_type() -> Response | tuple[Response, int]:
    """Update the robot-type label used in ``info.json`` and folder names.

    Refused mid-recording so the running session keeps a stable label. Persisted
    to gui_settings.json so it survives restarts.
    """
    if not ros_node:
        return jsonify({"ok": False, "msg": "ROS node not ready"}), 503
    data = request.get_json() or {}
    robot_type = str(data.get("robot_type", "")).strip()
    if not robot_type:
        return jsonify({"ok": False, "msg": "robot_type required"}), 400
    snap = state.snapshot()
    if snap.get("recorder_active"):
        return jsonify({
            "ok": False,
            "msg": "Stop the current recording before changing robot type.",
        }), 409
    with state.lock:
        state.robot_type = robot_type
    ros_node.publish_robot_type(robot_type)
    try:
        settings = _load_settings()
        settings["robot_type"] = robot_type
        _save_settings(settings)
    except Exception as exc:
        ros_node.get_logger().warn(f"Could not persist robot_type: {exc}")
    return jsonify({"ok": True, "robot_type": robot_type})


@app.route("/api/speed_scale", methods=["POST"])
def api_set_speed_scale() -> Response | tuple[Response, int]:
    """Set robot speed scale (0..1). Live-published on /gui/speed_scale and
    persisted to gui_settings.json so it survives restarts.
    """
    if not ros_node:
        return jsonify({"ok": False, "msg": "ROS node not ready"}), 503
    data = request.get_json() or {}
    try:
        scale = float(data.get("speed_scale"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "speed_scale required (float)"}), 400
    scale = max(0.01, min(1.0, scale))
    ros_node.publish_speed_scale(scale)
    # GR00T TCP-EE rollouts are absolute Cartesian setpoints — slowing them
    # requires stretching the publish cadence, not just clamping servoL velocity.
    # Forward the live scale to model_runner so the inference loop hz scales too.
    if model_runner is not None:
        try:
            model_runner.set_speed_scale(scale)
        except Exception as exc:
            ros_node.get_logger().warn(
                f"model_runner.set_speed_scale failed: {exc}")
    try:
        settings = _load_settings()
        settings["speed_scale"] = scale
        _save_settings(settings)
    except Exception as exc:
        ros_node.get_logger().warn(f"Could not persist speed_scale: {exc}")
    return jsonify({"ok": True, "speed_scale": scale})


@app.route("/api/confirm_motion", methods=["POST"])
def api_confirm_motion() -> Response | tuple[Response, int]:
    """Confirm motion — tells destination writer to connect and start."""
    if ros_node:
        data = request.get_json() or {}
        use_home = data.get("use_home", True)
        ros_node.publish_confirm_motion(use_home=use_home)
        return jsonify({"ok": True, "msg": "Motion confirmed"})
    return jsonify({"ok": False, "msg": "ROS node not ready"}), 503


@app.route("/api/recovery_ack", methods=["POST"])
def api_recovery_ack() -> Response | tuple[Response, int]:
    """Operator confirms the workspace is clear after a collision so
    destination_writer can unlock the protective stop and return to home. Safety
    interlock: arm will NOT move on its own without this.
    """
    if ros_node:
        ros_node.publish_recovery_ack()
        return jsonify({"ok": True, "msg": "Recovery acknowledged"})
    return jsonify({"ok": False, "msg": "ROS node not ready"}), 503


@app.route("/api/interp_scale", methods=["POST"])
def api_interp_scale() -> Response | tuple[Response, int]:
    """Set TCP interpolation horizon multiplier (0 = off, 1 = default)."""
    if ros_node is None:
        return jsonify({"ok": False, "msg": "ROS node not ready"}), 503
    data = request.get_json() or {}
    try:
        s = float(data.get("interp_scale"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "interp_scale must be a number"}), 400
    s = max(0.0, min(10.0, s))
    ros_node.publish_interp_scale(s)
    try:
        settings = _load_settings()
        settings["interp_scale"] = s
        _save_settings(settings)
    except Exception as exc:
        ros_node.get_logger().warn(f"Could not persist interp_scale: {exc}")
    return jsonify({"ok": True, "interp_scale": s})


@app.route("/api/model/infer_hz", methods=["POST"])
def api_model_infer_hz() -> Response | tuple[Response, int]:
    """Set the inference loop frequency live. Persists to gui_settings.json."""
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    data = request.get_json() or {}
    try:
        hz = float(data.get("hz"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "hz must be a number"}), 400
    res = model_runner.set_inference_hz(hz)
    if res.get("ok"):
        try:
            settings = _load_settings()
            settings["model_inference_hz"] = res.get("hz")
            _save_settings(settings)
        except Exception as exc:
            if ros_node is not None:
                ros_node.get_logger().warn(
                    f"Could not persist model_inference_hz: {exc}")
    return jsonify(res)


# ── Camera MJPEG streams ────────────────────────────────────────────────────

# Pre-rendered "No signal" placeholders, one per stream. Generated lazily on
# first use and cached.
_PLACEHOLDER_CACHE: dict = {}


def _placeholder_jpeg(label: str) -> bytes:
    if label in _PLACEHOLDER_CACHE:
        return _PLACEHOLDER_CACHE[label]
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (24, 24, 32)
    cv2.putText(img, "NO SIGNAL", (110, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (90, 90, 110), 3, cv2.LINE_AA)
    cv2.putText(img, label, (110, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 140, 160), 2, cv2.LINE_AA)
    cv2.putText(img, "check USB / serial", (110, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 130), 1, cv2.LINE_AA)
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    out = jpeg.tobytes()
    _PLACEHOLDER_CACHE[label] = out
    return out


_STREAM_LABELS = {
    "color": "Camera 1 - RGB",
    "depth": "Camera 1 - Depth",
    "color2": "Camera 2 - RGB",
    "depth2": "Camera 2 - Depth",
}


def _mjpeg_generator(which: str) -> Iterator[bytes]:
    """Yield MJPEG frames for the given camera stream."""
    label = _STREAM_LABELS.get(which, which)
    is_depth = which in ("depth", "depth2")
    while True:
        with state.lock:
            if is_depth and not state.show_depth:
                # User hid depth previews — serve a static placeholder so the
                # <img> doesn't show a broken icon.
                frame = _placeholder_jpeg(f"{label}\n(disabled)")
            elif which == "color":
                frame = state.color_frame
            elif which == "depth":
                frame = state.depth_frame
            elif which == "color2":
                frame = state.color_frame2
            elif which == "depth2":
                frame = state.depth_frame2
            else:
                frame = None
        if frame is None:
            frame = _placeholder_jpeg(label)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.066)  # ~15 fps


@app.route("/video/color")
def video_color() -> Response:
    return Response(_mjpeg_generator("color"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/depth")
def video_depth() -> Response:
    return Response(_mjpeg_generator("depth"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/color2")
def video_color2() -> Response:
    return Response(_mjpeg_generator("color2"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/depth2")
def video_depth2() -> Response:
    return Response(_mjpeg_generator("depth2"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Recordings ───────────────────────────────────────────────────────────────

def _bag_info(bag_path: Path) -> dict | None:
    """Extract info from a ROS bag's .db3 file."""
    db_files = list(bag_path.glob("*.db3"))
    if not db_files:
        return None
    try:
        conn = sqlite3.connect(str(db_files[0]))
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, type FROM topics")
        topics = cursor.fetchall()

        total_msgs = 0
        topic_details = []
        first_ts = None
        last_ts = None

        for tid, tname, ttype in topics:
            cursor.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM messages WHERE topic_id = ?", (tid,))
            count, tmin, tmax = cursor.fetchone()
            total_msgs += count
            topic_details.append({
                "name": tname,
                "type": ttype,
                "count": count,
            })
            if tmin is not None:
                if first_ts is None or tmin < first_ts:
                    first_ts = tmin
                if last_ts is None or tmax > last_ts:
                    last_ts = tmax

        conn.close()

        duration = (last_ts - first_ts) / 1e9 if first_ts and last_ts else 0
        return {
            "name": bag_path.name,
            "duration": round(duration, 1),
            "messages": total_msgs,
            "topics": topic_details,
            "topic_count": len(topics),
        }
    except Exception:
        return None


@app.route("/api/recordings")
def api_recordings() -> Response:
    """List all verified recordings."""
    recs = []
    rec_dir = Path(RECORDINGS_DIR)
    if rec_dir.exists():
        for d in sorted(rec_dir.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("recording_"):
                info = _bag_info(d)
                if info:
                    recs.append(info)
    return jsonify(recs)


@app.route("/api/recordings/<name>", methods=["DELETE"])
def api_delete_recording(name: str) -> Response | tuple[Response, int]:
    """Delete a recording."""
    bag_path = Path(RECORDINGS_DIR) / name
    if bag_path.exists() and bag_path.is_dir():
        shutil.rmtree(str(bag_path))
        return jsonify({"ok": True, "msg": f"Deleted {name}"})
    return jsonify({"ok": False, "msg": "Not found"}), 404


@app.route("/api/recordings/<name>/playback")
def api_playback(name: str) -> Response | tuple[Response, int]:
    """Extract RGB frames from a recording for video playback.

    Returns an MJPEG stream of the color topic.
    """
    bag_path = Path(RECORDINGS_DIR) / name
    if not bag_path.exists():
        return jsonify({"ok": False, "msg": "Not found"}), 404

    db_files = list(bag_path.glob("*.db3"))
    if not db_files:
        return jsonify({"ok": False, "msg": "No db3 file"}), 404

    def generate_playback() -> Iterator[bytes]:
        try:
            conn = sqlite3.connect(str(db_files[0]))
            cursor = conn.cursor()

            # Find color image topic id
            cursor.execute(
                "SELECT id FROM topics WHERE name = '/camera/camera/color/image_raw'")
            row = cursor.fetchone()
            if not row:
                return
            topic_id = row[0]

            cursor.execute(
                "SELECT data, timestamp FROM messages "
                "WHERE topic_id = ? ORDER BY timestamp", (topic_id,))

            prev_ts = None
            for data_blob, ts in cursor:
                # Deserialize ROS2 CDR Image → skip first 4 bytes (CDR header)
                try:
                    raw = bytes(data_blob)
                    # Parse CDR-encoded Image message
                    # Skip CDR encapsulation header (4 bytes)
                    offset = 4
                    # header.stamp.sec (4 bytes, uint32)
                    offset += 4
                    # header.stamp.nanosec (4 bytes, uint32)
                    offset += 4
                    # header.frame_id (4 bytes length + string + padding)
                    frame_id_len = int.from_bytes(raw[offset:offset + 4], "little")
                    offset += 4 + frame_id_len
                    # Align to 4 bytes
                    offset = (offset + 3) & ~3
                    # height (uint32)
                    height = int.from_bytes(raw[offset:offset + 4], "little")
                    offset += 4
                    # width (uint32)
                    width = int.from_bytes(raw[offset:offset + 4], "little")
                    offset += 4
                    # encoding (4 bytes length + string)
                    enc_len = int.from_bytes(raw[offset:offset + 4], "little")
                    offset += 4
                    encoding = raw[offset:offset + enc_len - 1].decode("utf-8")
                    offset += enc_len
                    # Align to 4 bytes
                    offset = (offset + 3) & ~3
                    # is_bigendian (uint8)
                    offset += 1
                    # Align to 4 bytes
                    offset = (offset + 3) & ~3
                    # step (uint32, unused)
                    offset += 4
                    # data (sequence: 4 bytes length + bytes)
                    data_len = int.from_bytes(raw[offset:offset + 4], "little")
                    offset += 4
                    img_data = raw[offset:offset + data_len]

                    if encoding == "rgb8":
                        img = np.frombuffer(img_data, dtype=np.uint8).reshape(
                            height, width, 3)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif encoding == "bgr8":
                        img = np.frombuffer(img_data, dtype=np.uint8).reshape(
                            height, width, 3)
                    else:
                        continue

                    _, jpeg = cv2.imencode(".jpg", img,
                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" +
                           jpeg.tobytes() + b"\r\n")

                    # Maintain original timing
                    if prev_ts is not None:
                        delay = (ts - prev_ts) / 1e9
                        delay = min(delay, 0.2)  # cap at 200ms
                        time.sleep(delay)
                    prev_ts = ts

                except Exception:
                    continue

            conn.close()
        except Exception:
            pass

    return Response(generate_playback(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Settings ─────────────────────────────────────────────────────────────────

# Settings are persisted to a JSON file
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "gui_settings.json")

DEFAULT_SETTINGS = {
    "source_ip": "192.168.1.103",
    "dest_ip": "192.168.1.102",
    "gripper_port": 63352,
    "gap_multiplier": 10.0,
    "min_gap_ms": 500.0,
    "grace_period": 2.0,
    "catch_up_speed": 0.1,
    "mode": "no-home",
    # When True, the next launch of run_recorder.sh starts the recorder with
    # record_tcp_pose:=true and record_depth:=true so the dataset captures the
    # source TCP pose + depth from both cameras (useful for retraining policies
    # on delta-EE actions).
    "record_extras": False,
    # Per-episode language instruction (task description). Used by SmolVLA / pi0
    # / GR00T as the text prompt during training and inference. Edit per session
    # in the GUI; persisted here so the field stays populated across restarts.
    "task_description": "Pick Large White Gear, Place Blue Bin",
    # Robot speed scale (0.01..1.0) applied to MIRRORING servoJ/servoL velocity +
    # acceleration. Default 10% for safe first-run policy rollouts. The GUI
    # slider live-publishes /gui/speed_scale and persists the value here so it
    # survives restarts.
    "speed_scale": 0.10,
}


def _load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        with contextlib.suppress(Exception):
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            # Merge with defaults
            merged = dict(DEFAULT_SETTINGS)
            merged.update(saved)
            return merged
    return dict(DEFAULT_SETTINGS)


def _save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


@app.route("/api/settings", methods=["GET"])
def api_get_settings() -> Response:
    return jsonify(_load_settings())


@app.route("/api/settings", methods=["POST"])
def api_set_settings() -> Response | tuple[Response, int]:
    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "msg": "No data"}), 400
    settings = _load_settings()
    for key in DEFAULT_SETTINGS:
        if key in data:
            settings[key] = data[key]
    _save_settings(settings)
    return jsonify({"ok": True, "settings": settings})


# ── AI Model control ────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def api_list_models() -> Response | tuple[Response, int]:
    """List available checkpoint versions under the model base dir."""
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    return jsonify({
        "ok": True,
        "base_dir": str(model_runner.base_dir),
        "versions": model_runner.list_versions(),
        "status": model_runner.status(),
    })


@app.route("/api/model/start", methods=["POST"])
def api_model_start() -> tuple[Response, int]:
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    data = request.get_json() or {}
    version = str(data.get("version", "")).strip()
    if not version:
        return jsonify({"ok": False, "msg": "version required"}), 400
    task = data.get("task")
    res = model_runner.start(version, task=task)
    if res.get("ok") and ros_node is not None:
        ros_node._publish_policy_active(True)
        # Seed the runner with the currently-persisted speed scale so GR00T
        # playback hz reflects the slider immediately on first rollout, without
        # waiting for the user to nudge the slider.
        try:
            persisted_ss = float(_load_settings().get("speed_scale", 1.0))
            model_runner.set_speed_scale(persisted_ss)
        except Exception as exc:
            ros_node.get_logger().warn(
                f"Could not seed model_runner speed_scale: {exc}")
    code = 200 if res.get("ok") else 409
    return jsonify(res), code


@app.route("/api/model/load", methods=["POST"])
def api_model_load() -> tuple[Response, int]:
    """Load (or swap to) a checkpoint without starting the inference loop.

    Robot is not driven; the policy stays warm in GPU until Start, Unload, or a
    subsequent Load.
    """
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    data = request.get_json() or {}
    version = str(data.get("version", "")).strip()
    if not version:
        return jsonify({"ok": False, "msg": "version required"}), 400
    task = data.get("task")
    res = model_runner.load(version, task=task, autostart=False)
    code = 200 if res.get("ok") else 409
    return jsonify(res), code


@app.route("/api/model/unload", methods=["POST"])
def api_model_unload() -> Response | tuple[Response, int]:
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    res = model_runner.unload()
    if ros_node is not None:
        ros_node._publish_policy_active(False)
    return jsonify(res)


@app.route("/api/model/task", methods=["POST"])
def api_model_task() -> Response | tuple[Response, int]:
    """Live-update the language task description used by GR00T inference.

    Takes effect on the next inference call; no reload required. The string is
    also persisted to gui_settings.json so it survives restarts.
    """
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    data = request.get_json() or {}
    task = str(data.get("task", "")).strip()
    if not task:
        return jsonify({"ok": False, "msg": "task required"}), 400
    res = model_runner.set_task(task)
    try:
        settings = _load_settings()
        settings["model_task"] = task
        _save_settings(settings)
    except Exception as exc:
        if ros_node is not None:
            ros_node.get_logger().warn(f"Could not persist model_task: {exc}")
    return jsonify(res)


@app.route("/api/model/chunk_steps", methods=["POST"])
def api_model_chunk_steps() -> Response | tuple[Response, int]:
    """Cap how many actions of each inferred chunk get executed.

    Smaller = more reactive but more compute. ``0`` or missing disables the cap
    (plays the full action_horizon). Persists to gui_settings.json.
    """
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    data = request.get_json() or {}
    raw = data.get("steps")
    try:
        n = int(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "steps must be an integer"}), 400
    if n is not None and n < 0:
        return jsonify({"ok": False, "msg": "steps must be >= 0"}), 400
    res = model_runner.set_max_steps_per_chunk(n)
    try:
        settings = _load_settings()
        settings["model_max_steps_per_chunk"] = res.get("max_steps_per_chunk")
        _save_settings(settings)
    except Exception as exc:
        if ros_node is not None:
            ros_node.get_logger().warn(
                f"Could not persist model_max_steps_per_chunk: {exc}")
    return jsonify(res)


@app.route("/api/model/ensemble", methods=["POST"])
def api_model_ensemble() -> Response | tuple[Response, int]:
    """Set the temporal-ensemble window K (number of recent chunks to
    weighted-average per tick). K=1 disables ensembling.
    """
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    data = request.get_json() or {}
    raw = data.get("window")
    try:
        k = int(raw) if raw not in (None, "") else 1
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "window must be an integer"}), 400
    if k < 1:
        return jsonify({"ok": False, "msg": "window must be >= 1"}), 400
    res = model_runner.set_ensemble_window(k)
    try:
        settings = _load_settings()
        settings["model_ensemble_window"] = res.get("ensemble_window")
        _save_settings(settings)
    except Exception as exc:
        if ros_node is not None:
            ros_node.get_logger().warn(
                f"Could not persist model_ensemble_window: {exc}")
    return jsonify(res)


@app.route("/api/model/stop", methods=["POST"])
def api_model_stop() -> Response | tuple[Response, int]:
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    res = model_runner.stop()
    if ros_node is not None:
        ros_node._publish_policy_active(False)
    return jsonify(res)


@app.route("/api/model/status", methods=["GET"])
def api_model_status() -> Response | tuple[Response, int]:
    if model_runner is None:
        return jsonify({"ok": False, "msg": "Model runner not ready"}), 503
    return jsonify(model_runner.status())


# ── Episode replay control ──────────────────────────────────────────

@app.route("/api/replay/sessions", methods=["GET"])
def api_replay_sessions() -> Response | tuple[Response, int]:
    """Enumerate sessions + episode lists for the replay UI."""
    if replay_runner is None:
        return jsonify({"ok": False, "msg": "Replay runner not ready"}), 503
    return jsonify({
        "ok": True,
        "recordings_dir": str(replay_runner.recordings_dir),
        "sessions": replay_runner.list_sessions(),
        "status": replay_runner.status(),
    })


@app.route("/api/replay/start", methods=["POST"])
def api_replay_start() -> Response | tuple[Response, int]:
    if replay_runner is None:
        return jsonify({"ok": False, "msg": "Replay runner not ready"}), 503
    data = request.get_json() or {}
    session = str(data.get("session", "")).strip()
    if not session:
        return jsonify({"ok": False, "msg": "session required"}), 400
    try:
        episode = int(data.get("episode", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "episode must be int"}), 400
    # ``mode`` selects joint replay (servoJ, no IK) vs TCP replay (servoL, UR
    # controller does the IK). Defaults to joints for backwards-compatible
    # behaviour.
    mode = str(data.get("mode", "joints")).lower()
    res = replay_runner.start(session, episode, mode=mode)
    if res.get("ok") and ros_node is not None:
        # Same gating as model start: tell destination_writer to enter MIRRORING
        # via /policy/active so it stays in policy mode for the duration of the
        # episode.
        ros_node._publish_policy_active(True)
    code = 200 if res.get("ok") else 409
    return jsonify(res), code


@app.route("/api/replay/stop", methods=["POST"])
def api_replay_stop() -> Response | tuple[Response, int]:
    if replay_runner is None:
        return jsonify({"ok": False, "msg": "Replay runner not ready"}), 503
    res = replay_runner.stop()
    if ros_node is not None:
        ros_node._publish_policy_active(False)
    return jsonify(res)


@app.route("/api/replay/status", methods=["GET"])
def api_replay_status() -> Response | tuple[Response, int]:
    if replay_runner is None:
        return jsonify({"ok": False, "msg": "Replay runner not ready"}), 503
    return jsonify(replay_runner.status())


# ── SocketIO events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect() -> None:
    socketio.emit("state_update", state.snapshot())


@socketio.on("request_toggle")
def on_toggle() -> None:
    if not ros_node:
        return
    snap = state.snapshot()
    # Apply the same readiness gate as /api/toggle so the DI0-equivalent socket
    # path can't bypass the check.
    if not snap.get("recorder_active") and not snap.get("ready_to_record", True):
        socketio.emit("toggle_blocked", {
            "blockers": snap.get("recording_blockers", []),
        })
        return
    ros_node.publish_toggle()


@socketio.on("request_new_session")
def on_new_session() -> None:
    if not ros_node:
        return
    snap = state.snapshot()
    if snap.get("recorder_active"):
        socketio.emit("new_session_blocked", {
            "msg": "Stop the current recording before starting a new session.",
        })
        return
    ros_node.publish_new_session()
    with state.lock:
        state.recording_count = 0
        state.grip_points = []


@socketio.on("request_confirm_motion")
def on_confirm_motion(data: dict | None = None) -> None:
    if ros_node:
        use_home = True
        use_depth = False
        if data and isinstance(data, dict):
            use_home = data.get("use_home", True)
            use_depth = bool(data.get("use_depth", False))
        # Persist the user's depth-preview choice so the MJPEG depth endpoints
        # can short-circuit when the user hid them.
        with state.lock:
            state.show_depth = use_depth
        ros_node.publish_confirm_motion(use_home=use_home)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: list[str] | None = None) -> None:
    global ros_node
    global model_runner
    global replay_runner

    rclpy.init(args=args)
    ros_node = GuiRosNode()

    # Seed task description from persisted settings and broadcast once so the
    # recorder picks it up via the TRANSIENT_LOCAL latch.
    try:
        persisted = _load_settings().get(
            "task_description", "Pick Large White Gear, Place Blue Bin")
        with state.lock:
            state.task_description = persisted
        ros_node.publish_task_description(persisted)
    except Exception as exc:
        ros_node.get_logger().warn(
            f"Could not seed task_description from settings: {exc}")

    # Seed robot type the same way.
    try:
        persisted_rt = _load_settings().get("robot_type", "UR10_Single")
        with state.lock:
            state.robot_type = persisted_rt
        ros_node.publish_robot_type(persisted_rt)
    except Exception as exc:
        ros_node.get_logger().warn(
            f"Could not seed robot_type from settings: {exc}")

    # Seed speed scale (latched, picked up by destination_writer whenever it
    # subscribes).
    try:
        persisted_ss = float(_load_settings().get("speed_scale", 0.10))
        ros_node.publish_speed_scale(persisted_ss)
    except Exception as exc:
        ros_node.get_logger().warn(
            f"Could not seed speed_scale from settings: {exc}")

    # Seed interpolation scale (latched). 1.0 = default behaviour.
    try:
        persisted_is = float(_load_settings().get("interp_scale", 1.0))
        ros_node.publish_interp_scale(persisted_is)
    except Exception as exc:
        ros_node.get_logger().warn(
            f"Could not seed interp_scale from settings: {exc}")

    # Wire up the AI model runner. It republishes /mirror/joint_states +
    # /mirror/gripper/position so the destination_writer state machine drives the
    # robot to the policy's predicted actions.
    model_runner = ModelRunner(
        base_dir=MODEL_BASE_DIR,
        publish_action=ros_node.publish_model_action,
        publish_tcp=ros_node.publish_model_tcp_action,
        logger=ros_node.get_logger(),
    )
    model_runner.set_observation_provider(_model_observation_provider)
    # Seed task string from persisted settings so the text-input box and the
    # runner agree out of the gate.
    try:
        persisted_settings = _load_settings()
        persisted_task = persisted_settings.get("model_task")
        if persisted_task:
            model_runner.set_task(str(persisted_task))
        persisted_cap = persisted_settings.get("model_max_steps_per_chunk")
        if persisted_cap is not None:
            model_runner.set_max_steps_per_chunk(persisted_cap)
        persisted_ensemble = persisted_settings.get("model_ensemble_window")
        if persisted_ensemble is not None:
            model_runner.set_ensemble_window(persisted_ensemble)
        persisted_hz = persisted_settings.get("model_inference_hz")
        if persisted_hz is not None:
            model_runner.set_inference_hz(persisted_hz)
    except Exception as exc:
        ros_node.get_logger().warn(
            f"Could not seed model settings: {exc}")

    # Wire up the episode replay runner. It republishes the same /mirror/* topics
    # so the destination_writer state machine drives the follower through a
    # previously recorded episode. When the worker finishes naturally (i.e. the
    # episode runs out), flip /policy/active back to False so destination_writer
    # returns home.
    replay_runner = ReplayRunner(
        recordings_dir=Path(RECORDINGS_LEROBOT_DIR),
        publish_action=ros_node.publish_model_action,
        publish_tcp=ros_node.publish_model_tcp_action,
        logger=ros_node.get_logger(),
    )

    def _on_replay_finished() -> None:
        if ros_node is not None:
            ros_node._publish_policy_active(False)

    replay_runner.set_on_finished(_on_replay_finished)

    # Spin ROS2 in a background thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,),
                                  daemon=True)
    ros_thread.start()

    # Run Flask+SocketIO (blocks)
    try:
        socketio.run(app, host="0.0.0.0", port=8080, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        if model_runner is not None:
            model_runner.stop()
        if replay_runner is not None:
            replay_runner.stop()
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
