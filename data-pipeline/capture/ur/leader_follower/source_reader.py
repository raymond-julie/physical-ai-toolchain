#!/usr/bin/env python3
"""Source robot reader for the UR leader/follower teleop+record tool.

Reads real-time joint positions via RTDE, gripper status via the Robotiq
socket protocol, and tool digital inputs (DI0, DI1). Publishes everything on
``/mirror/*`` topics consumed by the destination writer and the recorder.

Source robot IP default: 192.168.1.80

Requirements:
    pip install ur_rtde
"""

from __future__ import annotations

import contextlib
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64

try:
    from rtde_io import RTDEIOInterface
    from rtde_receive import RTDEReceiveInterface

    RTDE_AVAILABLE = True
except ImportError:  # pragma: no cover
    RTDE_AVAILABLE = False
    print("WARNING: ur_rtde not installed. Install with: pip install ur_rtde")


class RobotiqSocketClient:
    """Client for a Robotiq gripper via the raw socket protocol (port 63352)."""

    def __init__(self, robot_ip: str, port: int = 63352, timeout: float = 1.0) -> None:
        self.robot_ip = robot_ip
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.connected = False

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.robot_ip, self.port))
            self.connected = True
            return True
        except OSError:
            self.connected = False
            return False

    def disconnect(self) -> None:
        if self.sock:
            with contextlib.suppress(OSError):
                self.sock.close()
        self.connected = False

    def _send_command(self, cmd: str) -> str | None:
        if not self.connected or self.sock is None:
            return None
        try:
            self.sock.sendall(f"{cmd}\n".encode())
            return self.sock.recv(1024).decode().strip()
        except OSError:
            self.connected = False
            return None

    def get_position(self) -> int:
        resp = self._send_command("GET POS")
        if resp and resp.startswith("POS"):
            try:
                return int(resp.split()[1])
            except (IndexError, ValueError):
                return -1
        return -1

    def activate(self) -> None:
        self._send_command("SET ACT 1")

    def move_to(self, position: int, speed: int = 255, force: int = 50) -> bool:
        if not self.connected:
            return False
        try:
            self._send_command(f"SET POS {int(position)}")
            self._send_command(f"SET SPE {int(speed)}")
            self._send_command(f"SET FOR {int(force)}")
            self._send_command("SET GTO 1")
            return True
        except OSError:
            self.connected = False
            return False


class SourceReaderNode(Node):
    """Reads source robot state and publishes ``/mirror/*`` topics."""

    DEFAULT_ROBOT_IP = "192.168.1.80"
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    ROBOTIQ_PORT = 63352
    GRIPPER_CLOSED_THRESHOLD = 128

    def __init__(self) -> None:
        super().__init__("source_reader")

        self.declare_parameter("robot_ip", self.DEFAULT_ROBOT_IP)
        self.declare_parameter("rtde_frequency", 125.0)
        self.declare_parameter("gripper_closed_threshold", self.GRIPPER_CLOSED_THRESHOLD)

        self.robot_ip = self.get_parameter("robot_ip").value
        rtde_frequency = self.get_parameter("rtde_frequency").value
        self.gripper_threshold = self.get_parameter("gripper_closed_threshold").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # RELIABLE QoS for joint mirroring — BEST_EFFORT silently drops samples
        # under CPU/network pressure (cameras + bag recording on the same
        # machine make this worse), causing the destination to lag and then
        # catch up in a sudden jump.
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.current_joint_positions = [0.0] * 6
        self.current_joint_velocities = [0.0] * 6
        self.gripper_position_raw = 0
        self.gripper_position = 0.0
        self.gripper_is_closed = False
        self.tool_digital_in_0 = False
        self.tool_digital_in_1 = False
        self.tool_din_0_prev = False
        self.tool_din_1_prev = False
        self.tool_dout_0_state = False
        self.gripper_toggle_closed = False
        self.last_di0_toggle_time = 0.0
        self.last_di1_toggle_time = 0.0
        self.DEBOUNCE_SECONDS = 0.5
        self.rtde_connected = False
        self.io_connected = False
        self.gripper_connected = False
        self.rtde: RTDEReceiveInterface | None = None
        self.rtde_io: RTDEIOInterface | None = None
        self.robotiq: RobotiqSocketClient | None = None

        self.mirror_joint_pub = self.create_publisher(JointState, "/mirror/joint_states", joint_qos)
        self.mirror_gripper_position_pub = self.create_publisher(Float64, "/mirror/gripper/position", qos)
        self.mirror_gripper_status_pub = self.create_publisher(Bool, "/mirror/gripper/is_closed", qos)
        self.mirror_tool_din_0_pub = self.create_publisher(Bool, "/mirror/tool_digital_input_0", qos)
        self.mirror_tool_din_1_pub = self.create_publisher(Bool, "/mirror/tool_digital_input_1", qos)

        # Diagnostics for jerk debugging.
        self.publish_count = 0
        self.last_pub_positions: list[float] | None = None

        if RTDE_AVAILABLE:
            self.connect_rtde()
        else:
            self.get_logger().error("ur_rtde not available!")
        self.connect_gripper()

        self.create_timer(1.0 / rtde_frequency, self.read_and_publish)
        self.create_timer(0.05, self.read_gripper)
        self.create_timer(1.0, self.log_status)
        self.create_timer(2.0, self.check_connections)

        self.get_logger().info(f"Source Reader started — {self.robot_ip}")

    def connect_rtde(self) -> None:
        try:
            self.get_logger().info(f"Connecting RTDE to {self.robot_ip}...")
            self.rtde = RTDEReceiveInterface(self.robot_ip)
            self.rtde_connected = True
            self.get_logger().info("RTDE connection established!")
        except Exception as exc:
            self.get_logger().error(f"RTDE connection failed: {exc}")
            self.rtde_connected = False
        try:
            self.get_logger().info(f"Connecting RTDE IO to {self.robot_ip}...")
            self.rtde_io = RTDEIOInterface(self.robot_ip)
            self.io_connected = True
            self.get_logger().info("RTDE IO connected!")
            # Ensure DO0 (recording LED) is OFF on startup.
            try:
                self.rtde_io.setToolDigitalOut(0, False)
                self.tool_dout_0_state = False
                self.get_logger().info("DO0 (recording LED) set to OFF")
            except Exception as exc:
                self.get_logger().error(f"Failed to reset DO0: {exc}")
        except Exception as exc:
            self.get_logger().error(f"RTDE IO connection failed: {exc}")
            self.io_connected = False

    def connect_gripper(self) -> None:
        try:
            self.get_logger().info(f"Connecting Robotiq gripper at {self.robot_ip}:{self.ROBOTIQ_PORT}...")
            self.robotiq = RobotiqSocketClient(self.robot_ip, self.ROBOTIQ_PORT)
            if self.robotiq.connect():
                self.gripper_connected = True
                self.robotiq.activate()
                self.get_logger().info("Robotiq gripper connected and activated!")
            else:
                self.gripper_connected = False
        except Exception as exc:
            self.get_logger().error(f"Gripper connection failed: {exc}")
            self.gripper_connected = False

    def check_connections(self) -> None:
        if RTDE_AVAILABLE and not self.rtde_connected:
            self.connect_rtde()
        if RTDE_AVAILABLE and not self.io_connected:
            with contextlib.suppress(Exception):
                self.rtde_io = RTDEIOInterface(self.robot_ip)
                self.io_connected = True
        if not self.gripper_connected:
            self.connect_gripper()

    def read_gripper(self) -> None:
        if not self.gripper_connected or self.robotiq is None:
            return
        try:
            pos = self.robotiq.get_position()
            if pos >= 0:
                self.gripper_position_raw = pos
                self.gripper_position = pos / 255.0
                self.gripper_is_closed = pos > self.gripper_threshold
            else:
                self.gripper_connected = False
        except Exception:
            self.gripper_connected = False

    def read_and_publish(self) -> None:
        if not RTDE_AVAILABLE or not self.rtde_connected or self.rtde is None:
            return
        try:
            if not self.rtde.isConnected():
                self.rtde_connected = False
                return

            # Use TARGET (controller setpoint) instead of ACTUAL (encoder
            # feedback). getTargetQ is the smooth, internally-filtered value
            # the UR controller is commanding; getActualQ contains encoder
            # noise that, forwarded to the destination servoJ at high gain,
            # manifests as visible high-frequency vibration / jerk.
            try:
                self.current_joint_positions = list(self.rtde.getTargetQ())
            except Exception:
                self.current_joint_positions = list(self.rtde.getActualQ())
            try:
                self.current_joint_velocities = list(self.rtde.getTargetQd())
            except Exception:
                self.current_joint_velocities = list(self.rtde.getActualQd())

            digital_input_bits = self.rtde.getActualDigitalInputBits()
            self.tool_digital_in_0 = bool(digital_input_bits & (1 << 16))
            self.tool_digital_in_1 = bool(digital_input_bits & (1 << 17))

            now = time.monotonic()

            # DI0 rising edge → toggle tool DO0.
            if (
                self.tool_digital_in_0
                and not self.tool_din_0_prev
                and (now - self.last_di0_toggle_time) >= self.DEBOUNCE_SECONDS
            ):
                self.tool_dout_0_state = not self.tool_dout_0_state
                self.last_di0_toggle_time = now
                self.get_logger().info(f'DI0 rising edge — DO0 → {"ON" if self.tool_dout_0_state else "OFF"}')
                if self.io_connected and self.rtde_io is not None:
                    try:
                        self.rtde_io.setToolDigitalOut(0, self.tool_dout_0_state)
                    except Exception as exc:
                        self.get_logger().error(f"Failed to set tool DO0: {exc}")
                        self.io_connected = False
            self.tool_din_0_prev = self.tool_digital_in_0

            # DI1 rising edge → toggle gripper.
            if (
                self.tool_digital_in_1
                and not self.tool_din_1_prev
                and (now - self.last_di1_toggle_time) >= self.DEBOUNCE_SECONDS
            ):
                self.gripper_toggle_closed = not self.gripper_toggle_closed
                grip_target = 255 if self.gripper_toggle_closed else 0
                self.last_di1_toggle_time = now
                self.get_logger().info(
                    f'DI1 rising edge — gripper → {"CLOSED" if self.gripper_toggle_closed else "OPEN"}'
                )
                if self.gripper_connected and self.robotiq is not None:
                    try:
                        self.robotiq.move_to(grip_target, speed=255, force=50)
                    except Exception:
                        self.gripper_connected = False
            self.tool_din_1_prev = self.tool_digital_in_1

            jm = JointState()
            jm.header.stamp = self.get_clock().now().to_msg()
            jm.name = self.JOINT_NAMES
            jm.position = self.current_joint_positions
            jm.velocity = self.current_joint_velocities
            self.mirror_joint_pub.publish(jm)
            self.publish_count += 1

            gp = Float64()
            gp.data = self.gripper_position
            self.mirror_gripper_position_pub.publish(gp)

            gs = Bool()
            gs.data = self.gripper_is_closed
            self.mirror_gripper_status_pub.publish(gs)

            d0 = Bool()
            d0.data = self.tool_digital_in_0
            self.mirror_tool_din_0_pub.publish(d0)

            d1 = Bool()
            d1.data = self.tool_digital_in_1
            self.mirror_tool_din_1_pub.publish(d1)
        except Exception as exc:
            self.get_logger().error(f"RTDE read error: {exc}")
            self.rtde_connected = False

    def log_status(self) -> None:
        r = "OK" if self.rtde_connected else "DOWN"
        g = "OK" if self.gripper_connected else "DOWN"
        pos = ", ".join(f"{p:.3f}" for p in self.current_joint_positions)
        # Jerk diagnostics: actual publish rate, max velocity, max delta.
        max_vel = max((abs(v) for v in self.current_joint_velocities), default=0.0)
        max_delta = 0.0
        if self.last_pub_positions is not None and self.current_joint_positions:
            max_delta = max(abs(a - b) for a, b in zip(self.current_joint_positions, self.last_pub_positions))
        self.last_pub_positions = list(self.current_joint_positions)
        self.get_logger().info(
            f"RTDE[{r}] PubHz={self.publish_count} maxV={max_vel:.3f} maxD/s={max_delta:.3f} | "
            f"Joints: [{pos}] | "
            f'Gripper[{g}]: {self.gripper_position_raw}/255 ({"closed" if self.gripper_is_closed else "open"}) | '
            f'DI0:{"ON" if self.tool_digital_in_0 else "OFF"} DI1:{"ON" if self.tool_digital_in_1 else "OFF"} | '
            f'DO0:{"ON" if self.tool_dout_0_state else "OFF"} '
            f'Grip:{"CLOSED" if self.gripper_toggle_closed else "OPEN"}'
        )
        self.publish_count = 0

    def destroy_node(self) -> bool:
        if self.rtde is not None:
            with contextlib.suppress(Exception):
                self.rtde.disconnect()
        if self.rtde_io is not None:
            with contextlib.suppress(Exception):
                self.rtde_io.disconnect()
        if self.robotiq is not None:
            self.robotiq.disconnect()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    if not RTDE_AVAILABLE:
        print("ERROR: ur_rtde not installed!  pip install ur_rtde")
        return
    rclpy.init(args=args)
    node = SourceReaderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().info("Shutting down Source Reader")
    finally:
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
