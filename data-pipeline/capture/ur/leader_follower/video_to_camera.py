#!/usr/bin/env python3
"""Replay an mp4 (or a folder of images) on the RealSense camera topics.

Stand-in for ``ros2 launch realsense2_camera rs_launch.py`` when no
physical Intel RealSense devices are connected. Publishes the same
topic names that ``lerobot_recorder_node.py`` subscribes to so the
recorder cannot tell the difference between a live camera and a
replayed video file.

Typical use cases
-----------------
* Lab cameras are down and the operator still wants to exercise the
  recorder pipeline (parquet rows, mp4 chunks, ACSA blob mirroring).
* CI / smoke test of the LeRobotDataset writer without USB hardware.
* Replaying a previously recorded LeRobot mp4 (e.g.
  ``videos/chunk-000/observation.images.color/episode_000000.mp4``) as
  if it were a live feed — the recorder then round-trips the same
  pixels through encode + ACSA upload.

Topic mapping
-------------
* ``camera_name=camera1`` (default) publishes:
    ``/camera1/camera1/color/image_raw``
* ``camera_name=camera2`` publishes:
    ``/camera2/camera2/color/image_raw``

These topic names match the defaults declared in
``lerobot_recorder_node.py`` (``color_topic`` / ``color_topic2``).

Depth is not synthesised; the recorder ignores depth when
``RECORD_DEPTH=false`` (the default for ``run_recorder.sh``).

Parameters
----------
* ``video_path`` (string, required unless ``synthetic=True``):
    Path to an mp4 (or any file ``cv2.VideoCapture`` can open).
* ``camera_name`` (string, default ``camera1``):
    Namespace and node-internal name. Set to ``camera2`` for the
    second simulated camera.
* ``fps`` (double, default ``15.0``):
    Publish rate in Hz. Independent of the file's encoded fps; we just
    pull the next frame on each timer tick.
* ``loop`` (bool, default ``True``):
    When True, restart from frame 0 on EOF.
* ``synthetic`` (bool, default ``False``):
    When True, ignore ``video_path`` and publish a 640x480 moving-bars
    test pattern (useful when no mp4 is available at all).
* ``width`` / ``height`` (int, default ``640`` / ``480``):
    Only used in ``synthetic`` mode.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    print("ERROR: opencv-python missing. Install with:\n  pip install opencv-python", file=sys.stderr)
    raise

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover
    print("ERROR: cv_bridge missing. Install with:\n  sudo apt install ros-${ROS_DISTRO}-cv-bridge", file=sys.stderr)
    raise


class VideoSourceError(RuntimeError):
    """Raised when a configured video source is missing or cannot be opened."""


class VideoToCamera(Node):
    """Publishes mp4 frames (or a synthetic pattern) as ``sensor_msgs/Image``."""

    def __init__(self) -> None:
        super().__init__("video_to_camera")

        self.declare_parameter("video_path", "")
        self.declare_parameter("camera_name", "camera1")
        # Declared as DOUBLE so the launcher can pass either ``15`` or
        # ``15.0`` without ROS 2 Jazzy raising InvalidParameterTypeException.
        from rcl_interfaces.msg import ParameterDescriptor, ParameterType

        self.declare_parameter(
            "fps",
            15.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, dynamic_typing=True),
        )
        self.declare_parameter("loop", True)
        self.declare_parameter("synthetic", False)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)

        self._video_path: str = self.get_parameter("video_path").value
        self._camera_name: str = self.get_parameter("camera_name").value
        self._fps: float = float(self.get_parameter("fps").value)
        self._loop: bool = bool(self.get_parameter("loop").value)
        self._synthetic: bool = bool(self.get_parameter("synthetic").value)
        self._width: int = int(self.get_parameter("width").value)
        self._height: int = int(self.get_parameter("height").value)

        if not self._synthetic:
            if not self._video_path:
                raise VideoSourceError("video_path is required when synthetic=False")
            if not os.path.isfile(self._video_path):
                raise VideoSourceError(f"video_path does not exist: {self._video_path}")

        # Match the realsense2_camera namespacing convention:
        # /<camera_namespace>/<camera_name>/color/image_raw
        topic = f"/{self._camera_name}/{self._camera_name}/color/image_raw"

        # Match the recorder's image QoS exactly so subscription
        # negotiation succeeds without warnings.
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._pub = self.create_publisher(Image, topic, img_qos)
        self._bridge = CvBridge()
        self._frame_idx = 0
        self._tick = 0

        if self._synthetic:
            self._cap = None
            self.get_logger().info(
                f"Publishing synthetic {self._width}x{self._height} test "
                f"pattern at {self._fps:.1f} Hz on {topic}"
            )
        else:
            self._cap = cv2.VideoCapture(self._video_path)
            if not self._cap.isOpened():
                raise VideoSourceError(f"cv2.VideoCapture failed to open {self._video_path}")
            n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps = self._cap.get(cv2.CAP_PROP_FPS) or float("nan")
            self.get_logger().info(
                f"Replaying {self._video_path} ({n_frames} frames @ "
                f"{src_fps:.1f} fps) at {self._fps:.1f} Hz on {topic} (loop={self._loop})"
            )

        self._timer = self.create_timer(1.0 / max(self._fps, 0.1), self._on_tick)

    def _on_tick(self) -> None:
        if self._synthetic:
            frame = self._make_synthetic_frame()
        else:
            ok, frame = self._cap.read()
            if not ok:
                if not self._loop:
                    self.get_logger().info("End of video — shutting down.")
                    rclpy.shutdown()
                    return
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    self.get_logger().error("Failed to read frame after seek.")
                    return
            self._frame_idx += 1

        msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{self._camera_name}_color_optical_frame"
        self._pub.publish(msg)
        self._tick += 1

    def _make_synthetic_frame(self) -> np.ndarray:
        """Build a moving vertical-bar pattern so the feed is visibly live."""
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        bar_width = max(self._width // 16, 8)
        offset = (self._tick * 4) % (bar_width * 2)
        for x in range(-offset, self._width, bar_width * 2):
            cv2.rectangle(
                frame,
                (x, 0),
                (min(x + bar_width, self._width), self._height),
                (40, 200, 240),  # warm yellow stripes on black
                thickness=-1,
            )
        cv2.putText(
            frame,
            f"{self._camera_name} SYN {self._tick:06d}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def destroy_node(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        return super().destroy_node()


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = VideoToCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
