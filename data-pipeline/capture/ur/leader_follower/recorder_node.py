#!/usr/bin/env python3
"""Legacy ROS 2 bag recorder node for the UR leader/follower tool.

Subscribes to ``/recorder/active`` (published by the destination writer):

* When True  → start ROS bag recording.
* When False → stop recording and verify the bag for timestamp continuity.

Clean bags are moved to ``./recordings/``; bags with gaps are deleted. The
LeRobot writer (``lerobot_recorder_node.py``) supersedes this node for normal
operation; it is retained for raw-topic capture and debugging.

Requirements:
    ROS 2 Humble, Python >= 3.10
"""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_bag_recorder import RosBagRecorder
from std_msgs.msg import Bool


class RecorderNode(Node):
    """Listens to ``/recorder/active`` and manages ROS bag recording."""

    def __init__(self) -> None:
        super().__init__("recorder_node")

        self.declare_parameter("output_dir", "./recordings_raw")
        self.declare_parameter("validated_dir", "./recordings")
        self.declare_parameter("gap_multiplier", 10.0)
        self.declare_parameter("grace_period", 2.0)
        self.declare_parameter("min_gap_ms", 500.0)

        self.output_dir = self.get_parameter("output_dir").value
        self.validated_dir = self.get_parameter("validated_dir").value
        self.gap_multiplier = self.get_parameter("gap_multiplier").value
        self.grace_period = self.get_parameter("grace_period").value
        self.min_gap_ms = self.get_parameter("min_gap_ms").value

        self.topics_to_record = [
            "/camera/camera/color/image_raw",
            "/camera/camera/depth/image_rect_raw",
            "/joint_states",
            "/mirror/gripper/position",
            "/mirror/gripper/is_closed",
        ]

        self.recorder = RosBagRecorder(output_dir=self.output_dir)
        self.is_recording = False
        self.current_bag_name: str | None = None
        self.recording_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Bool, "/recorder/active", self._active_cb, qos)
        self.create_timer(5.0, self._log_tick)

        self.get_logger().info("Recorder Node started")
        self.get_logger().info(f"Raw bags  → {self.output_dir}")
        self.get_logger().info(f"Verified  → {self.validated_dir}")
        self.get_logger().info(f"Topics: {self.topics_to_record}")

    def _active_cb(self, msg: Bool) -> None:
        if msg.data and not self.is_recording:
            self._start_recording()
        elif not msg.data and self.is_recording:
            self._stop_recording()

    def _start_recording(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_bag_name = f"recording_{timestamp}"
        self.recorder.bag_name = self.current_bag_name

        self.get_logger().info(f"🔴 RECORDING STARTED: {self.current_bag_name}")
        self.recorder.start_recording(topics=self.topics_to_record)
        self.is_recording = True
        self.recording_count += 1

    def _stop_recording(self) -> None:
        self.get_logger().info(f"⏹ RECORDING STOPPED: {self.current_bag_name}")
        self.recorder.stop_recording()
        self.is_recording = False

        bag_path = Path(self.output_dir) / self.current_bag_name
        if bag_path.exists():
            self._verify_bag(bag_path)
        else:
            self.get_logger().warn(f"Bag directory not found: {bag_path}")

        self.current_bag_name = None

    def _verify_bag(self, bag_path: Path) -> None:
        """Verify a bag's timestamp continuity, then move or delete it."""
        db_files = list(bag_path.glob("*.db3"))
        if not db_files:
            self.get_logger().warn(f"No .db3 in {bag_path.name}")
            return

        db_path = db_files[0]
        has_gaps = False

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            self.get_logger().info(f"📁 Verifying: {bag_path.name}")

            cursor.execute("SELECT id, name, type FROM topics")
            topics = cursor.fetchall()

            for topic_id, topic_name, _ in topics:
                cursor.execute(
                    "SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
                    (topic_id,),
                )
                timestamps = [row[0] for row in cursor.fetchall()]

                if not timestamps:
                    self.get_logger().warn(f"  ⚠️  {topic_name}: NO messages")
                    has_gaps = True
                    continue

                if len(timestamps) < 2:
                    self.get_logger().info(f"  ℹ️  {topic_name}: {len(timestamps)} msg(s)")
                    continue

                ts_sec = [t / 1e9 for t in timestamps]

                # Skip messages within the grace period (startup jitter).
                t0 = ts_sec[0]
                grace_idx = 0
                for j, t in enumerate(ts_sec):
                    if t - t0 >= self.grace_period:
                        grace_idx = j
                        break
                else:
                    self.get_logger().info(
                        f"  ℹ️  {topic_name}: {len(timestamps)} msgs (all within grace period)"
                    )
                    continue

                ts_after_grace = ts_sec[grace_idx:]
                intervals = [
                    ts_after_grace[i] - ts_after_grace[i - 1] for i in range(1, len(ts_after_grace))
                ]

                if not intervals:
                    self.get_logger().info(
                        f"  ℹ️  {topic_name}: {len(timestamps)} msgs (only 1 msg after grace period)"
                    )
                    continue

                avg_interval = sum(intervals) / len(intervals)
                max_gap = max(intervals)
                avg_hz = 1.0 / avg_interval if avg_interval > 0 else 0
                threshold = max(avg_interval * self.gap_multiplier, self.min_gap_ms / 1000.0)
                gaps = [g for g in intervals if g > threshold]

                if gaps:
                    self.get_logger().warn(
                        f"  ⚠️  {topic_name}: {len(timestamps)} msgs, {avg_hz:.1f} Hz, "
                        f"MAX GAP {max_gap * 1000:.0f}ms ({len(gaps)} gaps)"
                    )
                    has_gaps = True
                else:
                    self.get_logger().info(
                        f"  ✅ {topic_name}: {len(timestamps)} msgs, {avg_hz:.1f} Hz, "
                        f"max gap {max_gap * 1000:.0f}ms"
                    )

            conn.close()
        except Exception as exc:
            self.get_logger().error(f"Verification error: {exc}")
            return

        if has_gaps:
            try:
                shutil.rmtree(str(bag_path))
                self.get_logger().warn(f"  🗑️  DELETED {bag_path.name} (had gaps)")
            except OSError as exc:
                self.get_logger().error(f"Delete failed: {exc}")
        else:
            validated = Path(self.validated_dir)
            validated.mkdir(parents=True, exist_ok=True)
            dest = validated / bag_path.name
            try:
                shutil.move(str(bag_path), str(dest))
                self.get_logger().info(f"  🎉 MOVED {bag_path.name} → recordings/")
            except OSError as exc:
                self.get_logger().error(f"Move failed: {exc}")

    def _log_tick(self) -> None:
        status = "RECORDING" if self.is_recording else "IDLE"
        bag = self.current_bag_name or "—"
        self.get_logger().info(f"[{status}] Bag: {bag} | Total recordings: {self.recording_count}")

    def destroy_node(self) -> bool:
        if self.is_recording:
            self.get_logger().info("Shutdown — stopping active recording")
            self.recorder.stop_recording()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().info("Shutting down Recorder Node")
    finally:
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
