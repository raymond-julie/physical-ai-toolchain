#!/usr/bin/env python3
"""Replay a recorded LeRobot episode on the follower (destination) robot.

Streams the recorded ``action[0:6]`` joint vector to ``/mirror/joint_states``
and the ``action.gripper.position`` analog encoder feedback (0..1) to
``/mirror/gripper/position`` at the dataset fps. The destination_writer node is
responsible for actually moving the UR + Robotiq.

Prerequisites (verified by the script before publishing anything):

1. ``destination_writer.py`` is running (subscribes to /mirror/*).
2. ``source_reader.py`` is NOT running (would race for /mirror/*).
3. ``/destination/state`` reports IDLE (i.e. the user has confirmed motion via
   the GUI and the follower has aligned to home).

Once those are satisfied the script:

* sends a one-shot rising-edge on /mirror/tool_digital_input_0 to transition the
  destination_writer state machine IDLE → MIRRORING
* streams the recorded action vector at the dataset fps
* sends a second rising-edge to transition MIRRORING → RETURNING

Usage::

    python3 replay_episode.py --session <session_dir> [--episode N] \
                              [--fps 15] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String

_LOGGER = logging.getLogger(__name__)

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class ReplayNode(Node):
    def __init__(self, fps: float) -> None:
        super().__init__("replay_episode")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Match the topics destination_writer.py subscribes to.
        self.pub_joints = self.create_publisher(
            JointState, "/mirror/joint_states", qos)
        self.pub_gripper = self.create_publisher(
            Float64, "/mirror/gripper/position", qos)
        self.pub_di0 = self.create_publisher(
            Bool, "/mirror/tool_digital_input_0", qos)
        # Health subscriptions used to verify destination_writer is alive and in
        # the right state before we start streaming.
        self._dest_state: str | None = None
        self.create_subscription(
            String, "/destination/state",
            lambda m: setattr(self, "_dest_state", m.data), qos)
        self.fps = fps
        self.dt = 1.0 / fps

    # ── helpers ────────────────────────────────────────────────
    def wait_for_state(self, expected: str, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._dest_state == expected:
                return True
        return False

    def publish_joints(self, q: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [float(x) for x in q[:6]]
        self.pub_joints.publish(msg)

    def publish_gripper(self, pos: float) -> None:
        m = Float64()
        m.data = float(pos)
        self.pub_gripper.publish(m)

    def pulse_di0(self) -> None:
        """One-shot rising edge on /mirror/tool_digital_input_0.

        destination_writer's debounce wants the level to actually toggle
        False → True, so we send False first then True a few ticks later.
        """
        for _ in range(3):
            m = Bool()
            m.data = False
            self.pub_di0.publish(m)
            time.sleep(0.05)
        for _ in range(5):
            m = Bool()
            m.data = True
            self.pub_di0.publish(m)
            time.sleep(0.05)


def load_episode(session_dir: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    parquet_files = sorted((session_dir / "data").rglob("file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {session_dir}/data")
    df = pq.read_table(parquet_files[0]).to_pandas()
    if episode not in df.episode_index.unique():
        raise ValueError(
            f"Episode {episode} not in dataset; available: "
            f"{sorted(df.episode_index.unique())}")
    ep = df[df.episode_index == episode].sort_values("frame_index")
    actions = np.stack(ep["action"].values).astype(np.float32)  # (N, 7)
    grip = ep["action.gripper.position"].values.astype(np.float32)  # (N,)
    return actions, grip


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True, type=Path,
                    help="LeRobot session directory")
    ap.add_argument("--episode", type=int, default=0,
                    help="episode_index to replay (default: 0)")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="replay rate Hz (default: 15)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load + validate the episode but do not publish.")
    ap.add_argument("--skip-state-check", action="store_true",
                    help="Skip the /destination/state == IDLE precondition "
                         "(useful if destination_writer is older and does not "
                         "publish state).")
    args = ap.parse_args()

    actions, grip = load_episode(args.session, args.episode)
    n = len(actions)
    _LOGGER.info("[replay] loaded session=%s ep=%s frames=%s duration=%.2fs",
                 args.session.name, args.episode, n, n / args.fps)
    _LOGGER.info("[replay] first joints: %s", actions[0, :6])
    _LOGGER.info("[replay] first grip:   %.4f", grip[0])
    _LOGGER.info("[replay] last  joints: %s", actions[-1, :6])

    if args.dry_run:
        _LOGGER.info("[replay] --dry-run; not publishing")
        return

    rclpy.init()
    node = ReplayNode(fps=args.fps)
    try:
        if not args.skip_state_check:
            _LOGGER.info("[replay] waiting for /destination/state == IDLE...")
            if not node.wait_for_state("IDLE", timeout=5.0):
                _LOGGER.error(
                    "[replay] ABORT — destination state is %r (need IDLE). "
                    "Confirm motion in the GUI and let it align to home, or "
                    "pass --skip-state-check.", node._dest_state)
                sys.exit(2)
            _LOGGER.info("[replay] destination IDLE ✔")

        # Pre-stream the first joint sample so destination_writer's catch-up
        # phase can interpolate from current pose to the episode's start pose at
        # alignment_speed before we hit play.
        _LOGGER.info("[replay] sending start-pose hold for 1.5s (destination "
                     "will catch up at alignment_speed)...")
        for _ in range(int(1.5 * args.fps)):
            node.publish_joints(actions[0])
            node.publish_gripper(float(grip[0]))
            time.sleep(node.dt)

        _LOGGER.info("[replay] pulsing DI0 → MIRRORING")
        node.pulse_di0()
        # Give the state machine a moment to settle into MIRRORING.
        time.sleep(0.5)

        _LOGGER.info("[replay] streaming %s frames @ %s Hz...", n, args.fps)
        t0 = time.monotonic()
        for i in range(n):
            target = t0 + i * node.dt
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
            node.publish_joints(actions[i])
            node.publish_gripper(float(grip[i]))
            if i % 30 == 0:
                _LOGGER.info("  frame %4d/%d  q0..5=%s  grip=%.3f",
                             i, n, [round(float(v), 3) for v in actions[i, :6]],
                             grip[i])
        wall = time.monotonic() - t0
        _LOGGER.info("[replay] streamed in %.2fs (expected %.2fs)",
                     wall, n / args.fps)

        # Hold the final pose briefly so the follower fully tracks in.
        for _ in range(int(0.5 * args.fps)):
            node.publish_joints(actions[-1])
            node.publish_gripper(float(grip[-1]))
            time.sleep(node.dt)

        _LOGGER.info("[replay] pulsing DI0 → RETURNING")
        node.pulse_di0()
        time.sleep(0.5)
        _LOGGER.info("[replay] done")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
