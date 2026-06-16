#!/usr/bin/env python3
"""Web-GUI trigger + live camera preview — fallback for installations
without a physical record button.

Serves a single-page web UI on (by default) ``http://0.0.0.0:8080``
with:

* a Record/Stop button that toggles ``/recorder/active`` (same
  semantics as :mod:`episode_recorder.nodes.trigger_tool_io`, so both
  triggers can coexist without state divergence), and
* live MJPEG previews of every configured camera, so operators can
  verify framing before hitting Record.

ROS callbacks decode each ``sensor_msgs/Image`` to BGR, downscale to
``preview_width``, JPEG-encode and cache the latest bytes under a
per-topic condition variable. Each ``/api/stream/<idx>`` request
streams the cached bytes as a ``multipart/x-mixed-replace`` MJPEG feed.

Requirements:
    pip install flask opencv-python
    sudo apt install ros-${ROS_DISTRO}-cv-bridge
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import rclpy
from flask import Flask, Response, jsonify, render_template, request
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float64, String

# Resolve template/static dirs relative to this file so the node works
# regardless of where it is launched from.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEB_DIR = os.path.join(_PKG_DIR, "web")

app = Flask(
    __name__,
    template_folder=os.path.join(_WEB_DIR, "templates"),
    static_folder=os.path.join(_WEB_DIR, "static"),
)


class _CameraSlot:
    """Latest JPEG bytes + condition variable for one camera topic."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.last_update: float = 0.0  # monotonic seconds


class _SharedState:
    """Thread-safe glue between Flask handlers and the ROS node."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: bool = False
        self.pub = None  # rclpy publisher; set by node on init
        self.cameras: list[_CameraSlot] = []
        # Per-namespace robot state cache: {ns: {...}}
        self.robots: dict[str, dict[str, Any]] = {}
        # Latest recorder stats (parsed JSON from /recorder/stats), or None.
        self.stats: dict[str, Any] | None = None
        self.stats_updated: float = 0.0  # monotonic seconds

    def publish(self, value: bool) -> None:
        if self.pub is None:
            return
        msg = Bool()
        msg.data = bool(value)
        self.pub.publish(msg)


STATE = _SharedState()


# Lazy-imported on first use so the module still imports on machines
# without opencv (the node will then refuse to start, but tooling such
# as the registry autoload doesn't break).
_cv2 = None
_np = None
_bridge = None
_placeholder_jpeg: bytes | None = None


def _lazy_cv() -> None:
    global _cv2, _np, _bridge, _placeholder_jpeg
    if _cv2 is not None:
        return
    import cv2 as _c
    import numpy as _n

    try:
        from cv_bridge import CvBridge
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "cv_bridge missing. Install with: sudo apt install ros-${ROS_DISTRO}-cv-bridge"
        ) from exc
    _cv2 = _c
    _np = _n
    _bridge = CvBridge()
    img = _np.full((240, 320, 3), 32, dtype=_np.uint8)
    _cv2.putText(
        img, "no signal", (60, 130), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, _cv2.LINE_AA
    )
    ok, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 70])
    _placeholder_jpeg = buf.tobytes() if ok else b""


# ── Flask routes ────────────────────────────────────────────────


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        cameras=STATE.cameras,
        robots=sorted(STATE.robots.keys()),
    )


@app.route("/api/state")
def api_state() -> Response:
    with STATE.lock:
        active = STATE.active
        stats = dict(STATE.stats) if STATE.stats is not None else None
        stats_age = time.monotonic() - STATE.stats_updated if STATE.stats_updated else None
        robots_snap = {ns: dict(rs) for ns, rs in STATE.robots.items()}
    cams = []
    now = time.monotonic()
    for i, slot in enumerate(STATE.cameras):
        # A camera is considered "live" if a frame arrived in the last 2 s.
        age = (now - slot.last_update) if slot.last_update else None
        cams.append(
            {
                "index": i,
                "topic": slot.topic,
                "live": bool(slot.jpeg is not None and age is not None and age < 2.0),
                "age_seconds": round(age, 2) if age is not None else None,
            }
        )
    # Convert robot age timestamps from monotonic to age-in-seconds.
    robots_out = []
    for ns in sorted(robots_snap.keys()):
        rs = robots_snap[ns]
        j_age = (now - rs["joints_t"]) if rs.get("joints_t") else None
        g_age = (now - rs["gripper_t"]) if rs.get("gripper_t") else None
        robots_out.append(
            {
                "namespace": ns,
                "joints": rs.get("joints"),
                "joints_age_seconds": round(j_age, 2) if j_age is not None else None,
                "joints_live": bool(j_age is not None and j_age < 2.0),
                "gripper_position": rs.get("gripper_position"),
                "gripper_closed": rs.get("gripper_closed"),
                "gripper_age_seconds": round(g_age, 2) if g_age is not None else None,
                "gripper_live": bool(g_age is not None and g_age < 2.0),
            }
        )
    return jsonify(
        active=active,
        cameras=cams,
        robots=robots_out,
        stats=stats,
        stats_age_seconds=(round(stats_age, 2) if stats_age is not None else None),
    )


@app.route("/api/toggle", methods=["POST"])
def api_toggle() -> Response:
    with STATE.lock:
        new_val = not STATE.active
    STATE.publish(new_val)
    return jsonify(requested=new_val)


@app.route("/api/set", methods=["POST"])
def api_set() -> Response:
    raw = request.json.get("value") if request.is_json else request.form.get("value", "")  # type: ignore
    desired = str(raw).strip().lower() in ("1", "true", "on", "yes")
    STATE.publish(desired)
    return jsonify(requested=desired)


@app.route("/api/stream/<int:idx>")
def api_stream(idx: int) -> Response | tuple[str, int]:
    if idx < 0 or idx >= len(STATE.cameras):
        return ("Not Found", 404)
    slot = STATE.cameras[idx]

    def gen() -> Iterator[bytes]:
        last_seen = 0.0
        # Cap stream rate at ~15 FPS to keep CPU + network sane when
        # several browsers (or several panels) are watching at once.
        min_period = 1.0 / 15.0
        while True:
            with slot.cond:
                slot.cond.wait_for(
                    lambda last_seen=last_seen: slot.jpeg is not None
                    and slot.last_update > last_seen,
                    timeout=1.0,
                )
                jpeg = slot.jpeg
                last_seen = slot.last_update
            if jpeg is None:
                placeholder = _placeholder_jpeg or b""
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
                )
                time.sleep(0.5)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(min_period)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── ROS node ────────────────────────────────────────────────────


class GuiTriggerNode(Node):
    """Bridges Flask handlers to ``/recorder/active`` + camera previews."""

    def __init__(self) -> None:
        super().__init__("trigger_gui")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        # Empty default — an empty/blank string is filtered out below
        # so the node still works without any cameras.
        self.declare_parameter("image_topics", [""])
        # Robot namespaces to subscribe to for the status tiles. An
        # empty/blank entry is filtered out, so the default works even
        # when no robots are configured.
        self.declare_parameter("robot_namespaces", [""])
        self.declare_parameter("preview_width", 480)
        self.declare_parameter("preview_jpeg_quality", 70)

        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        raw_topics = self.get_parameter("image_topics").value or []
        self.image_topics: list[str] = [str(t).strip() for t in raw_topics if str(t).strip()]
        raw_ns = self.get_parameter("robot_namespaces").value or []
        self.robot_namespaces: list[str] = [
            str(n).strip().strip("/") for n in raw_ns if str(n).strip()
        ]
        self.preview_width = int(self.get_parameter("preview_width").value)
        self.jpeg_q = int(self.get_parameter("preview_jpeg_quality").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Separate callback groups so the heavy image-decode callbacks
        # do NOT serialize behind each other (or behind the recorder
        # status / joint-state callbacks) under MultiThreadedExecutor.
        # Each image topic gets its own MutuallyExclusiveCallbackGroup
        # -> up to N image threads run in parallel, one per camera,
        # so 4 cameras can stream simultaneously. The light-weight
        # control/state callbacks share a single group.
        self._control_cbg = MutuallyExclusiveCallbackGroup()
        # /recorder/active: TRANSIENT_LOCAL so the latest press is
        # delivered to the recorder when it subscribes, even if the
        # click happened during the recorder's startup warmup.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        STATE.pub = self.create_publisher(Bool, "/recorder/active", latched_qos)
        self.create_subscription(
            Bool, "/recorder/active", self._on_active, latched_qos, callback_group=self._control_cbg
        )

        # Recorder status JSON
        self.create_subscription(
            String, "/recorder/stats", self._on_stats, qos, callback_group=self._control_cbg
        )

        # Per-robot state (joints, gripper)
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        for ns in self.robot_namespaces:
            STATE.robots[ns] = {
                "joints": None,
                "joints_t": 0.0,
                "gripper_position": None,
                "gripper_closed": None,
                "gripper_t": 0.0,
            }
            setattr(
                self,
                f"_sub_js_{ns}",
                self.create_subscription(
                    JointState,
                    f"/{ns}/joint_states",
                    self._make_joints_cb(ns),
                    be,
                    callback_group=self._control_cbg,
                ),
            )
            setattr(
                self,
                f"_sub_gp_{ns}",
                self.create_subscription(
                    Float64,
                    f"/{ns}/gripper/position",
                    self._make_gpos_cb(ns),
                    be,
                    callback_group=self._control_cbg,
                ),
            )
            setattr(
                self,
                f"_sub_gc_{ns}",
                self.create_subscription(
                    Bool,
                    f"/{ns}/gripper/is_closed",
                    self._make_gclosed_cb(ns),
                    be,
                    callback_group=self._control_cbg,
                ),
            )

        if self.image_topics:
            _lazy_cv()
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=2,
            )
            for topic in self.image_topics:
                slot = _CameraSlot(topic)
                STATE.cameras.append(slot)
                # One callback group PER camera so the multi-threaded
                # executor can decode + JPEG-encode all cameras in
                # parallel. Without this the four image callbacks
                # serialize on a single thread and the GUI shows
                # one stream "live" while the others stall.
                cam_cbg = MutuallyExclusiveCallbackGroup()
                setattr(self, f"_cbg_{len(STATE.cameras)}", cam_cbg)
                # Keep a reference to the subscription on self so the
                # garbage collector doesn't drop it.
                setattr(
                    self,
                    f"_sub_{len(STATE.cameras)}",
                    self.create_subscription(
                        Image, topic, self._make_image_cb(slot), img_qos, callback_group=cam_cbg
                    ),
                )

    def _on_active(self, msg: Bool) -> None:
        with STATE.lock:
            STATE.active = bool(msg.data)

    def _on_stats(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with STATE.lock:
            STATE.stats = payload
            STATE.stats_updated = time.monotonic()

    def _make_joints_cb(self, ns: str) -> Callable[[JointState], None]:
        def cb(msg: JointState) -> None:
            positions = [float(p) for p in (msg.position or [])]
            with STATE.lock:
                rs = STATE.robots.setdefault(ns, {})
                rs["joints"] = positions
                rs["joints_t"] = time.monotonic()

        return cb

    def _make_gpos_cb(self, ns: str) -> Callable[[Float64], None]:
        def cb(msg: Float64) -> None:
            with STATE.lock:
                rs = STATE.robots.setdefault(ns, {})
                rs["gripper_position"] = float(msg.data)
                rs["gripper_t"] = time.monotonic()

        return cb

    def _make_gclosed_cb(self, ns: str) -> Callable[[Bool], None]:
        def cb(msg: Bool) -> None:
            with STATE.lock:
                rs = STATE.robots.setdefault(ns, {})
                rs["gripper_closed"] = bool(msg.data)
                rs["gripper_t"] = time.monotonic()

        return cb

    def _make_image_cb(self, slot: _CameraSlot) -> Callable[[Image], None]:
        """Build a per-topic Image callback bound to the given slot."""
        target_w = self.preview_width
        jpeg_q = self.jpeg_q
        node = self

        def cb(msg: Image) -> None:
            try:
                img = _bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                node.get_logger().warn(f"preview decode failed ({slot.topic}): {e}")
                return

            if img.shape[1] > target_w:
                scale = target_w / float(img.shape[1])
                new_h = max(1, round(img.shape[0] * scale))
                img = _cv2.resize(img, (target_w, new_h), interpolation=_cv2.INTER_AREA)

            ok, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            if not ok:
                return
            jpeg_bytes = buf.tobytes()

            with slot.cond:
                slot.jpeg = jpeg_bytes
                slot.last_update = time.monotonic()
                slot.cond.notify_all()

        return cb


def _run_flask(host: str, port: int) -> None:
    # Werkzeug's reloader spawns a child process — disable it because
    # we are inside a ROS node. Silence the per-request access log so
    # MJPEG streams don't drown the ROS log.
    import logging

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GuiTriggerNode()
    flask_thread = threading.Thread(target=_run_flask, args=(node.host, node.port), daemon=True)
    flask_thread.start()
    node.get_logger().info(
        f"GUI trigger ready on http://{node.host}:{node.port}  "
        f"cameras={len(node.image_topics)} ({node.image_topics})"
    )
    # One executor thread per camera + a few spare for control / state
    # callbacks. Single-threaded spin causes the JPEG-encoding image
    # callbacks to serialize so only one MJPEG stream visibly updates.
    n_cams = max(1, len(node.image_topics))
    executor = MultiThreadedExecutor(num_threads=n_cams + 4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
