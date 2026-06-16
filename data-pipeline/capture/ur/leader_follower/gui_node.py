#!/usr/bin/env python3
"""Web GUI node for the UR leader/follower teleop+record tool.

A Flask + SocketIO dashboard served as a ROS 2 node. It provides:

* Real-time robot state via WebSocket.
* Live camera preview via MJPEG streams.
* GUI-triggered start/stop (publishes ``/gui/toggle``).
* Recording list, playback, and deletion.
* Runtime settings management.

Serves on http://0.0.0.0:8080.

Requirements:
    pip install flask flask-socketio
"""

from __future__ import annotations

import contextlib
import json
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
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "templates")
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, "recordings")
RECORDINGS_RAW_DIR = os.path.join(SCRIPT_DIR, "recordings_raw")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["SECRET_KEY"] = "ur-recorder-gui"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


class GuiState:
    """Thread-safe state container shared between ROS 2 and Flask."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

        # Source robot.
        self.source_joints = [0.0] * 6
        self.source_gripper_pos = 0.0
        self.source_gripper_closed = False
        self.source_di0 = False
        self.source_di1 = False
        self.source_connected = False
        self.source_last_msg_time = 0.0

        # Destination robot.
        self.dest_joints = [0.0] * 6
        self.dest_gripper_pos = 0.0
        self.dest_connected = False
        self.dest_last_msg_time = 0.0

        # State machine.
        self.state = "WAITING"
        self.recorder_active = False
        self.motion_confirmed = False

        # Camera 1.
        self.color_frame = None  # latest JPEG bytes
        self.depth_frame = None  # latest JPEG bytes (colorized)
        self.camera_connected = False
        self.camera_last_msg_time = 0.0

        # Camera 2.
        self.color_frame2 = None
        self.depth_frame2 = None
        self.camera2_connected = False
        self.camera2_last_msg_time = 0.0

        # Recording.
        self.current_bag_name = None
        self.recording_start_time = None
        self.recording_count = 0

        # ``show_depth`` controls whether the GUI serves the depth MJPEG
        # streams. Initialised from the launcher's RECORD_DEPTH_DEFAULT env var
        # so it matches the launch flag; the user can flip it at confirm time.
        self.show_depth = os.environ.get("RECORD_DEPTH_DEFAULT", "false").lower() == "true"

    def snapshot(self) -> dict:
        """Return a JSON-serialisable dict of all state."""
        with self.lock:
            now = time.time()
            return {
                "source": {
                    "joints": [round(j, 4) for j in self.source_joints],
                    "gripper_pos": round(self.source_gripper_pos, 3),
                    "gripper_closed": self.source_gripper_closed,
                    "di0": self.source_di0,
                    "di1": self.source_di1,
                    "connected": (now - self.source_last_msg_time) < 3.0
                    if self.source_last_msg_time > 0
                    else False,
                },
                "dest": {
                    "joints": [round(j, 4) for j in self.dest_joints],
                    "gripper_pos": round(self.dest_gripper_pos, 3),
                    "connected": (now - self.dest_last_msg_time) < 3.0
                    if self.dest_last_msg_time > 0
                    else False,
                },
                "state": self.state,
                "motion_confirmed": self.motion_confirmed,
                "recorder_active": self.recorder_active,
                "camera_connected": (now - self.camera_last_msg_time) < 3.0
                if self.camera_last_msg_time > 0
                else False,
                "camera2_connected": (now - self.camera2_last_msg_time) < 3.0
                if self.camera2_last_msg_time > 0
                else False,
                "camera_model": os.environ.get("CAM1_MODEL", "") or None,
                "camera2_model": os.environ.get("CAM2_MODEL", "") or None,
                "show_depth": self.show_depth,
                "current_bag": self.current_bag_name,
                "recording_elapsed": round(now - self.recording_start_time, 1)
                if self.recording_start_time
                else None,
                "recording_count": self.recording_count,
            }


state = GuiState()


class GuiRosNode(Node):
    """ROS 2 node that bridges topics to the Flask GUI."""

    def __init__(self) -> None:
        super().__init__("gui_node")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Source robot.
        self.create_subscription(JointState, "/mirror/joint_states", self._src_joint_cb, qos)
        self.create_subscription(Float64, "/mirror/gripper/position", self._src_grip_pos_cb, qos)
        self.create_subscription(Bool, "/mirror/gripper/is_closed", self._src_grip_closed_cb, qos)
        self.create_subscription(Bool, "/mirror/tool_digital_input_0", self._src_di0_cb, qos)
        self.create_subscription(Bool, "/mirror/tool_digital_input_1", self._src_di1_cb, qos)

        # Destination robot.
        self.create_subscription(JointState, "/joint_states", self._dst_joint_cb, qos)
        self.create_subscription(Float64, "/destination/gripper/position", self._dst_grip_pos_cb, qos)
        self.create_subscription(String, "/destination/state", self._dest_state_cb, qos)

        # Recorder.
        self.create_subscription(Bool, "/recorder/active", self._recorder_active_cb, qos)

        # Camera 1.
        self.create_subscription(Image, "/camera1/camera1/color/image_raw", self._color_cb, qos)
        self.create_subscription(Image, "/camera1/camera1/depth/image_rect_raw", self._depth_cb, qos)

        # Camera 2.
        self.create_subscription(Image, "/camera2/camera2/color/image_raw", self._color2_cb, qos)
        self.create_subscription(Image, "/camera2/camera2/depth/image_rect_raw", self._depth2_cb, qos)

        # Publishers (GUI → destination writer).
        self.gui_toggle_pub = self.create_publisher(Bool, "/gui/toggle", qos)
        self.gui_confirm_pub = self.create_publisher(Bool, "/gui/confirm_motion", qos)
        self.gui_use_home_pub = self.create_publisher(Bool, "/gui/use_home", qos)

        # Periodic WebSocket push.
        self.create_timer(0.1, self._push_state)  # 10 Hz

        self.get_logger().info("GUI Node started — http://0.0.0.0:8080")

    def _src_joint_cb(self, msg: JointState) -> None:
        with state.lock:
            state.source_joints = list(msg.position[:6])
            state.source_last_msg_time = time.time()

    def _src_grip_pos_cb(self, msg: Float64) -> None:
        with state.lock:
            state.source_gripper_pos = msg.data

    def _src_grip_closed_cb(self, msg: Bool) -> None:
        with state.lock:
            state.source_gripper_closed = msg.data

    def _src_di0_cb(self, msg: Bool) -> None:
        with state.lock:
            state.source_di0 = msg.data

    def _src_di1_cb(self, msg: Bool) -> None:
        with state.lock:
            state.source_di1 = msg.data

    def _dst_joint_cb(self, msg: JointState) -> None:
        with state.lock:
            state.dest_joints = list(msg.position[:6])
            state.dest_last_msg_time = time.time()

    def _dst_grip_pos_cb(self, msg: Float64) -> None:
        with state.lock:
            state.dest_gripper_pos = msg.data

    def _dest_state_cb(self, msg: String) -> None:
        with state.lock:
            state.state = msg.data
            if msg.data != "WAITING":
                state.motion_confirmed = True

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

    @staticmethod
    def _encode_color(msg: Image) -> bytes | None:
        if msg.encoding == "rgb8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "bgr8":
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        else:
            return None
        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes()

    @staticmethod
    def _encode_depth(msg: Image) -> bytes | None:
        if msg.encoding != "16UC1":
            return None
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        _, jpeg = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes()

    def _color_cb(self, msg: Image) -> None:
        try:
            jpeg = self._encode_color(msg)
            if jpeg is None:
                return
            with state.lock:
                state.color_frame = jpeg
                state.camera_connected = True
                state.camera_last_msg_time = time.time()
        except Exception:
            pass

    def _depth_cb(self, msg: Image) -> None:
        try:
            jpeg = self._encode_depth(msg)
            if jpeg is None:
                return
            with state.lock:
                state.depth_frame = jpeg
        except Exception:
            pass

    def _color2_cb(self, msg: Image) -> None:
        try:
            jpeg = self._encode_color(msg)
            if jpeg is None:
                return
            with state.lock:
                state.color_frame2 = jpeg
                state.camera2_connected = True
                state.camera2_last_msg_time = time.time()
        except Exception:
            pass

    def _depth2_cb(self, msg: Image) -> None:
        try:
            jpeg = self._encode_depth(msg)
            if jpeg is None:
                return
            with state.lock:
                state.depth_frame2 = jpeg
        except Exception:
            pass

    def _push_state(self) -> None:
        with contextlib.suppress(Exception):
            socketio.emit("state_update", state.snapshot(), namespace="/")

    def publish_toggle(self) -> None:
        """Publish a Bool(True) pulse on ``/gui/toggle``."""
        msg = Bool()
        msg.data = True
        self.gui_toggle_pub.publish(msg)
        self.get_logger().info("GUI toggle published")

    def publish_confirm_motion(self, use_home: bool = True) -> None:
        """Publish the use_home choice, then confirm motion."""
        home_msg = Bool()
        home_msg.data = use_home
        self.gui_use_home_pub.publish(home_msg)
        self.get_logger().info(f"GUI use_home={use_home} published")

        msg = Bool()
        msg.data = True
        self.gui_confirm_pub.publish(msg)
        self.get_logger().info("GUI motion confirmation published")
        with state.lock:
            state.motion_confirmed = True


ros_node: GuiRosNode | None = None


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/state")
def api_state() -> Response:
    return jsonify(state.snapshot())


@app.route("/api/toggle", methods=["POST"])
def api_toggle() -> Response | tuple[Response, int]:
    """Software-trigger mirroring/recording toggle (same as a DI0 press)."""
    if ros_node:
        ros_node.publish_toggle()
        return jsonify({"ok": True, "msg": "Toggle published"})
    return jsonify({"ok": False, "msg": "ROS node not ready"}), 503


@app.route("/api/confirm_motion", methods=["POST"])
def api_confirm_motion() -> Response | tuple[Response, int]:
    """Confirm motion — tells the destination writer to connect and start."""
    if ros_node:
        data = request.get_json() or {}
        use_home = data.get("use_home", True)
        ros_node.publish_confirm_motion(use_home=use_home)
        return jsonify({"ok": True, "msg": "Motion confirmed"})
    return jsonify({"ok": False, "msg": "ROS node not ready"}), 503


# Pre-rendered "No signal" placeholders, one per stream, cached lazily.
_PLACEHOLDER_CACHE: dict[str, bytes] = {}


def _placeholder_jpeg(label: str) -> bytes:
    if label in _PLACEHOLDER_CACHE:
        return _PLACEHOLDER_CACHE[label]
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (24, 24, 32)
    cv2.putText(img, "NO SIGNAL", (110, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (90, 90, 110), 3, cv2.LINE_AA)
    cv2.putText(img, label, (110, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 140, 160), 2, cv2.LINE_AA)
    cv2.putText(
        img, "check USB / serial", (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 130), 1, cv2.LINE_AA
    )
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
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.066)  # ~15 fps


@app.route("/video/color")
def video_color() -> Response:
    return Response(_mjpeg_generator("color"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/depth")
def video_depth() -> Response:
    return Response(_mjpeg_generator("depth"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/color2")
def video_color2() -> Response:
    return Response(_mjpeg_generator("color2"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/depth2")
def video_depth2() -> Response:
    return Response(_mjpeg_generator("depth2"), mimetype="multipart/x-mixed-replace; boundary=frame")


def _bag_info(bag_path: Path) -> dict | None:
    """Extract summary info from a ROS bag's .db3 file."""
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
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id = ?",
                (tid,),
            )
            count, tmin, tmax = cursor.fetchone()
            total_msgs += count
            topic_details.append({"name": tname, "type": ttype, "count": count})
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
    """Stream RGB frames from a recording's color topic as MJPEG."""
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

            cursor.execute("SELECT id FROM topics WHERE name = '/camera/camera/color/image_raw'")
            row = cursor.fetchone()
            if not row:
                return
            topic_id = row[0]

            cursor.execute(
                "SELECT data, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,),
            )

            prev_ts = None
            for data_blob, ts in cursor:
                # Parse the CDR-encoded Image message by hand: skip the 4-byte
                # encapsulation header, then walk the fixed-layout fields.
                try:
                    raw = bytes(data_blob)
                    offset = 4
                    offset += 4  # header.stamp.sec (uint32)
                    offset += 4  # header.stamp.nanosec (uint32)
                    frame_id_len = int.from_bytes(raw[offset : offset + 4], "little")
                    offset += 4 + frame_id_len
                    offset = (offset + 3) & ~3  # align to 4 bytes
                    height = int.from_bytes(raw[offset : offset + 4], "little")
                    offset += 4
                    width = int.from_bytes(raw[offset : offset + 4], "little")
                    offset += 4
                    enc_len = int.from_bytes(raw[offset : offset + 4], "little")
                    offset += 4
                    encoding = raw[offset : offset + enc_len - 1].decode("utf-8")
                    offset += enc_len
                    offset = (offset + 3) & ~3  # align to 4 bytes
                    offset += 1  # is_bigendian (uint8)
                    offset = (offset + 3) & ~3  # align to 4 bytes
                    int.from_bytes(raw[offset : offset + 4], "little")  # step (uint32)
                    offset += 4
                    data_len = int.from_bytes(raw[offset : offset + 4], "little")
                    offset += 4
                    img_data = raw[offset : offset + data_len]

                    if encoding == "rgb8":
                        img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif encoding == "bgr8":
                        img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
                    else:
                        continue

                    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"

                    # Maintain original timing (capped at 200 ms).
                    if prev_ts is not None:
                        delay = min((ts - prev_ts) / 1e9, 0.2)
                        time.sleep(delay)
                    prev_ts = ts
                except Exception:
                    continue

            conn.close()
        except Exception:
            pass

    return Response(generate_playback(), mimetype="multipart/x-mixed-replace; boundary=frame")


# Settings are persisted to a JSON file.
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "gui_settings.json")

DEFAULT_SETTINGS = {
    "source_ip": "192.168.1.80",
    "dest_ip": "192.168.1.90",
    "gripper_port": 63352,
    "gap_multiplier": 10.0,
    "min_gap_ms": 500.0,
    "grace_period": 2.0,
    "catch_up_speed": 0.1,
    "mode": "no-home",
}


def _load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            merged = dict(DEFAULT_SETTINGS)
            merged.update(saved)
            return merged
        except Exception:
            pass
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


@socketio.on("connect")
def on_connect() -> None:
    socketio.emit("state_update", state.snapshot())


@socketio.on("request_toggle")
def on_toggle() -> None:
    if ros_node:
        ros_node.publish_toggle()


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


def main(args: list[str] | None = None) -> None:
    global ros_node

    rclpy.init(args=args)
    ros_node = GuiRosNode()

    # Spin ROS 2 in a background thread.
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    # Run Flask + SocketIO (blocks).
    try:
        socketio.run(app, host="0.0.0.0", port=8080, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
