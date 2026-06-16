#!/usr/bin/env python3
"""Generic Robot Reader Node — episode_recorder.

Driver-pluggable, read-only ROS 2 publisher for one robot + one gripper.
The robot vendor and gripper vendor are selected via the
``robot_driver`` / ``gripper_driver`` parameters. To support a new
vendor, write a driver under :mod:`episode_recorder.drivers` and
register it — no changes to this node are required.

Published topics (with ``name=robot1``)::

    /robot1/joint_states                 sensor_msgs/JointState
    /robot1/gripper/position             std_msgs/Float64  (0.0 .. 1.0)
    /robot1/gripper/is_closed            std_msgs/Bool
    /robot1/digital_input/<input_name>   std_msgs/Bool   (one per DI exposed)

This node never commands the robot. Optional digital-output driving
(e.g. a "REC" LED) is delegated to the driver via
``set_digital_output`` and is invoked by trigger nodes, not by this
reader.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64

from episode_recorder.drivers.registry import create_gripper_driver, create_state_driver


class RobotReaderNode(Node):
    """ROS 2 wrapper around a :class:`RobotStateDriver` + :class:`GripperDriver`."""

    def __init__(self) -> None:
        super().__init__("robot_reader")

        # Identity
        self.declare_parameter("name", "robot1")
        self.declare_parameter("rate_hz", 125.0)
        self.declare_parameter("gripper_rate_hz", 20.0)

        # Driver selection
        self.declare_parameter("robot_driver", "ur_rtde")
        self.declare_parameter("gripper_driver", "robotiq_socket")

        # Driver config (common keys; drivers ignore those they don't use).
        self.declare_parameter("robot_ip", "192.168.1.80")
        self.declare_parameter("robot_port", 0)
        self.declare_parameter("gripper_ip", "")  # empty -> fall back to robot_ip
        self.declare_parameter("gripper_port", 0)
        self.declare_parameter("gripper_closed_threshold", 128)

        # Nova driver kwargs (ignored by other drivers via **_kwargs).
        self.declare_parameter("nats_url", "")
        self.declare_parameter("nova_cell", "cell")
        self.declare_parameter("nova_controller", "")
        self.declare_parameter("nats_user", "")
        self.declare_parameter("nats_password", "")
        self.declare_parameter("nats_creds_file", "")

        self.name = str(self.get_parameter("name").value).strip("/")
        rate_hz = float(self.get_parameter("rate_hz").value)
        gripper_rate_hz = float(self.get_parameter("gripper_rate_hz").value)

        robot_driver = str(self.get_parameter("robot_driver").value)
        gripper_driver = str(self.get_parameter("gripper_driver").value)

        robot_ip = str(self.get_parameter("robot_ip").value)
        robot_port = int(self.get_parameter("robot_port").value)
        gripper_ip = str(self.get_parameter("gripper_ip").value) or robot_ip
        gripper_port = int(self.get_parameter("gripper_port").value)
        gripper_closed = int(self.get_parameter("gripper_closed_threshold").value)

        nats_url = str(self.get_parameter("nats_url").value)
        nova_cell = str(self.get_parameter("nova_cell").value)
        nova_controller = str(self.get_parameter("nova_controller").value) or self.name
        nats_user = str(self.get_parameter("nats_user").value)
        nats_password = str(self.get_parameter("nats_password").value)
        nats_creds_file = str(self.get_parameter("nats_creds_file").value)

        # Build drivers
        robot_cfg: dict[str, Any] = {"host": robot_ip}
        if robot_port:
            robot_cfg["port"] = robot_port
        # Nova-specific kwargs — absorbed by **_kwargs on other drivers.
        if nats_url:
            robot_cfg["nats_url"] = nats_url
        robot_cfg["cell"] = nova_cell
        robot_cfg["controller"] = nova_controller
        if nats_user:
            robot_cfg["nats_user"] = nats_user
        if nats_password:
            robot_cfg["nats_password"] = nats_password
        if nats_creds_file:
            robot_cfg["nats_creds_file"] = nats_creds_file

        gripper_cfg: dict[str, Any] = {
            "host": gripper_ip,
            "closed_threshold": gripper_closed,
        }
        if gripper_port:
            gripper_cfg["port"] = gripper_port

        try:
            self.robot_drv = create_state_driver(robot_driver, **robot_cfg)
        except Exception as e:
            self.get_logger().error(f'Failed to construct robot driver "{robot_driver}": {e}')
            raise
        try:
            self.gripper_drv = create_gripper_driver(gripper_driver, **gripper_cfg)
        except Exception as e:
            self.get_logger().error(f'Failed to construct gripper driver "{gripper_driver}": {e}')
            raise

        # QoS
        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        base = f"/{self.name}"
        self.joint_pub = self.create_publisher(JointState, f"{base}/joint_states", rel)
        self.gripper_pos_pub = self.create_publisher(Float64, f"{base}/gripper/position", be)
        self.gripper_closed_pub = self.create_publisher(Bool, f"{base}/gripper/is_closed", be)

        # Per-DI publishers are created lazily as the driver reveals them.
        self._di_pubs: dict[str, Any] = {}
        self._di_base = f"{base}/digital_input"
        self._di_qos = be

        self._publish_count = 0
        self._last_state = None
        self._last_gripper = None

        # Connect (failures are recoverable via reconnect timer).
        self.robot_drv.connect()
        self.gripper_drv.connect()

        # Timers
        self.create_timer(1.0 / rate_hz, self._tick_robot)
        self.create_timer(1.0 / gripper_rate_hz, self._tick_gripper)
        self.create_timer(2.0, self._tick_reconnect)
        self.create_timer(1.0, self._tick_log)

        self.get_logger().info(
            f"RobotReader[{self.name}] started "
            f"robot_driver={robot_driver} gripper_driver={gripper_driver} "
            f"ip={robot_ip}"
        )

    # ── Timers ──────────────────────────────────────────────────

    def _tick_robot(self) -> None:
        st = self.robot_drv.read_state()
        if st is None:
            return
        self._last_state = st
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = st.joint_names
        msg.position = list(st.joint_positions)
        msg.velocity = list(st.joint_velocities)
        self.joint_pub.publish(msg)
        self._publish_count += 1

        for di_name, value in st.digital_inputs.items():
            pub = self._di_pubs.get(di_name)
            if pub is None:
                pub = self.create_publisher(Bool, f"{self._di_base}/{di_name}", self._di_qos)
                self._di_pubs[di_name] = pub
            b = Bool()
            b.data = bool(value)
            pub.publish(b)

    def _tick_gripper(self) -> None:
        gs = self.gripper_drv.read_state()
        if gs is None:
            return
        self._last_gripper = gs
        p = Float64()
        p.data = float(gs.position)
        self.gripper_pos_pub.publish(p)
        c = Bool()
        c.data = bool(gs.is_closed)
        self.gripper_closed_pub.publish(c)

    def _tick_reconnect(self) -> None:
        if not self.robot_drv.is_connected:
            self.robot_drv.connect()
        if not self.gripper_drv.is_connected:
            self.gripper_drv.connect()

    def _tick_log(self) -> None:
        st = self._last_state
        pos_str = "?"
        di_str = ""
        if st is not None:
            pos_str = ", ".join(f"{p:.3f}" for p in st.joint_positions)
            di_str = " ".join(f'{n}={"ON" if v else "OFF"}' for n, v in st.digital_inputs.items())
        g = self._last_gripper
        g_str = f'{g.position:.2f}({"closed" if g.is_closed else "open"})' if g is not None else "?"
        self.get_logger().info(
            f"[{self.name}] "
            f'Robot[{"OK" if self.robot_drv.is_connected else "DOWN"}] '
            f"PubHz={self._publish_count} | "
            f"Joints:[{pos_str}] | "
            f'Gripper[{"OK" if self.gripper_drv.is_connected else "DOWN"}]={g_str}'
            f'{(" | " + di_str) if di_str else ""}'
        )
        self._publish_count = 0

    # ── Cleanup ─────────────────────────────────────────────────

    def destroy_node(self) -> None:
        with contextlib.suppress(Exception):
            self.robot_drv.disconnect()
        with contextlib.suppress(Exception):
            self.gripper_drv.disconnect()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    try:
        node = RobotReaderNode()
    except Exception as e:
        print(f"RobotReader failed to start: {e}", file=sys.stderr)
        rclpy.shutdown()
        raise
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
