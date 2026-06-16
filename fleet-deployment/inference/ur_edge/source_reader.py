#!/usr/bin/env python3
"""Source robot reader for the UR edge runtime.

Reads real-time joint positions via RTDE, gripper status via the Robotiq socket,
and tool digital inputs (DI0, DI1). Publishes everything on ``/mirror/*`` topics.

Source robot IP: 192.168.1.103

Requirements::

    pip install ur_rtde
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray

_LOGGER = logging.getLogger(__name__)

try:
    from rtde_control import RTDEControlInterface
    from rtde_io import RTDEIOInterface
    from rtde_receive import RTDEReceiveInterface
    _RTDE_AVAILABLE = True
except ImportError:
    _RTDE_AVAILABLE = False
    _LOGGER.warning("ur_rtde not installed. Install with: pip install ur_rtde")


# ── Robotiq socket client ───────────────────────────────────────────────────

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


# ── ROS2 node ────────────────────────────────────────────────────────────────

class SourceReaderNode(Node):
    """Reads source robot state and publishes ``/mirror/*`` topics."""

    DEFAULT_ROBOT_IP = "192.168.1.103"
    JOINT_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    ROBOTIQ_PORT = 63352
    GRIPPER_CLOSED_THRESHOLD = 128

    # Same joint vector used by destination_writer.HOME_POSITIONS so the leader
    # and follower end every episode at the same pose.
    HOME_POSITIONS = [
        1.3399624824523926,    # shoulder_pan_joint
        -1.2604854863933106,   # shoulder_lift_joint
        1.8152335325824183,    # elbow_joint
        -2.3439699612059535,   # wrist_1_joint
        -1.5236032644854944,   # wrist_2_joint
        -0.24175817171205694,  # wrist_3_joint
    ]
    # moveJ velocity/accel (rad/s, rad/s²) for the post-recording auto-home.
    HOME_VELOCITY = 0.46875
    HOME_ACCEL = 0.46875

    def __init__(self) -> None:
        super().__init__("source_reader")

        self.declare_parameter("robot_ip", self.DEFAULT_ROBOT_IP)
        self.declare_parameter("rtde_frequency", 125.0)
        self.declare_parameter("gripper_closed_threshold", self.GRIPPER_CLOSED_THRESHOLD)
        # When True, engage teachMode() on /recorder/active rising edge and
        # moveJ to HOME on falling edge. Requires the source arm controller to
        # be in Remote Control mode.
        self.declare_parameter("auto_teach_mode", True)

        self.robot_ip = self.get_parameter("robot_ip").value
        rtde_frequency = self.get_parameter("rtde_frequency").value
        self.gripper_threshold = self.get_parameter("gripper_closed_threshold").value
        self.auto_teach_mode = self.get_parameter("auto_teach_mode").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        # Use RELIABLE QoS for joint mirroring — BEST_EFFORT silently drops
        # samples under CPU/network pressure (cameras + bag recording on the
        # same machine make this far worse), causing the destination to lag and
        # then catch up in a sudden jump.
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        # State
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
        self.control_connected = False
        self.gripper_connected = False
        self.rtde = None
        self.rtde_io = None
        self.rtde_c = None
        self.robotiq: RobotiqSocketClient | None = None
        # Recording-driven teachMode state machine.
        self.recorder_active = False
        self.teach_mode_active = False
        # Background worker for the post-recording moveJ-home (so the ROS
        # executor stays responsive while the arm moves).
        self._motion_lock = threading.Lock()
        self._motion_thread: threading.Thread | None = None

        # Publishers
        self.mirror_joint_pub = self.create_publisher(JointState, "/mirror/joint_states", joint_qos)
        self.mirror_gripper_position_pub = self.create_publisher(Float64, "/mirror/gripper/position", qos)
        self.mirror_gripper_status_pub = self.create_publisher(Bool, "/mirror/gripper/is_closed", qos)
        self.mirror_tool_din_0_pub = self.create_publisher(Bool, "/mirror/tool_digital_input_0", qos)
        self.mirror_tool_din_1_pub = self.create_publisher(Bool, "/mirror/tool_digital_input_1", qos)
        # Source TCP pose (6-vec: x,y,z,rx,ry,rz in base frame, metres/rad).
        # Used by the GUI to plot a gripper-close map.
        self.mirror_tcp_pose_pub = self.create_publisher(
            Float64MultiArray, "/mirror/tcp_pose", qos)

        # When an AI policy is driving the destination robot, suppress the
        # leader robot's mirror publishes so the destination_writer follows ONLY
        # the policy's joint targets.
        self.policy_active = False
        self._policy_log_state = False
        self.create_subscription(
            Bool, "/policy/active", self._policy_active_cb, qos)

        # Recording lifecycle: engage/disengage source teachMode.
        self.create_subscription(
            Bool, "/recorder/active", self._recorder_active_cb, qos)

        # Diagnostics for jerk debugging
        self.publish_count = 0
        self.last_pub_positions: list[float] | None = None

        # Connect
        if _RTDE_AVAILABLE:
            self.connect_rtde()
        else:
            self.get_logger().error("ur_rtde not available!")
        self.connect_gripper()

        # Timers
        self.create_timer(1.0 / rtde_frequency, self.read_and_publish)
        self.create_timer(0.05, self.read_gripper)
        self.create_timer(1.0, self.log_status)
        self.create_timer(2.0, self.check_connections)

        # Loud one-line status so the operator can tell at startup whether
        # freedrive-on-record will work.
        teach_state = (
            "ARMED (will engage on /recorder/active rising edge)"
            if (self.auto_teach_mode and self.control_connected)
            else (
                "DISARMED — auto_teach_mode=False"
                if not self.auto_teach_mode
                else "DISARMED — RTDE Control not connected "
                     "(pendant in Local mode? will retry every 2s)"))
        self.get_logger().info(
            f"Source Reader started — {self.robot_ip}  "
            f"auto teachMode: {teach_state}")

    # ── connections ──────────────────────────────────────────────

    def connect_rtde(self) -> None:
        try:
            self.get_logger().info(f"Connecting RTDE to {self.robot_ip}...")
            self.rtde = RTDEReceiveInterface(self.robot_ip)
            self.rtde_connected = True
            self.get_logger().info("RTDE connection established!")
        except Exception as e:
            self.get_logger().error(f"RTDE connection failed: {e}")
            self.rtde_connected = False
        if self.auto_teach_mode:
            try:
                self.get_logger().info(
                    f"Connecting RTDE Control to {self.robot_ip} "
                    "(required for auto teachMode)...")
                self.rtde_c = RTDEControlInterface(self.robot_ip)
                self.control_connected = True
                self.get_logger().info("RTDE Control connected!")
            except Exception as e:
                self.get_logger().warn(
                    f"RTDE Control connection failed: {e}. "
                    "Source arm probably in Local mode — auto teachMode "
                    "disabled for this run.")
                self.control_connected = False
                self.rtde_c = None
        try:
            self.get_logger().info(f"Connecting RTDE IO to {self.robot_ip}...")
            self.rtde_io = RTDEIOInterface(self.robot_ip)
            self.io_connected = True
            self.get_logger().info("RTDE IO connected!")
            # Ensure DO0 (recording LED) is OFF on startup
            try:
                self.rtde_io.setToolDigitalOut(0, False)
                self.tool_dout_0_state = False
                self.get_logger().info("DO0 (recording LED) set to OFF")
            except Exception as e:
                self.get_logger().error(f"Failed to reset DO0: {e}")
        except Exception as e:
            self.get_logger().error(f"RTDE IO connection failed: {e}")
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
        except Exception as e:
            self.get_logger().error(f"Gripper connection failed: {e}")
            self.gripper_connected = False

    def check_connections(self) -> None:
        if _RTDE_AVAILABLE and not self.rtde_connected:
            self.connect_rtde()
        if _RTDE_AVAILABLE and not self.io_connected:
            with contextlib.suppress(Exception):
                self.rtde_io = RTDEIOInterface(self.robot_ip)
                self.io_connected = True
        # Retry RTDE Control independently: connect_rtde() only re-runs when
        # Receive is also down, so a pendant that booted in Local mode (Control
        # refused) would never recover even after the operator switches to
        # Remote. Without this, auto teachMode silently stops engaging on
        # /recorder/active.
        if (_RTDE_AVAILABLE and self.auto_teach_mode
                and not self.control_connected):
            self._try_connect_control(quiet=True)
        if not self.gripper_connected:
            self.connect_gripper()

    def _try_connect_control(self, quiet: bool = False) -> bool:
        """Attempt to (re)open the RTDE Control interface to the source arm.

        Returns True on success. Safe to call repeatedly.
        """
        if not _RTDE_AVAILABLE:
            return False
        try:
            self.rtde_c = RTDEControlInterface(self.robot_ip)
            self.control_connected = True
            self.get_logger().info(
                "RTDE Control (re)connected — auto teachMode armed.")
            return True
        except Exception as e:
            self.control_connected = False
            self.rtde_c = None
            if not quiet:
                self.get_logger().warn(
                    f"RTDE Control connect failed: {e} "
                    "(source pendant in Local mode?).")
            return False

    # ── gripper ──────────────────────────────────────────────────

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

    # ── main read + publish loop ─────────────────────────────────

    def read_and_publish(self) -> None:
        if not _RTDE_AVAILABLE or not self.rtde_connected or self.rtde is None:
            return
        try:
            if not self.rtde.isConnected():
                self.rtde_connected = False
                return

            # Use TARGET (controller setpoint) instead of ACTUAL (encoder
            # feedback). getTargetQ is the smooth, internally-filtered value the
            # UR controller is commanding; getActualQ contains encoder
            # noise/quantisation that, when forwarded to destination's servoJ at
            # high gain, manifests as visible high-frequency vibration / jerk.
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

            # DI0 rising edge → toggle tool DO0
            if (self.tool_digital_in_0 and not self.tool_din_0_prev
                    and (now - self.last_di0_toggle_time) >= self.DEBOUNCE_SECONDS):
                self.tool_dout_0_state = not self.tool_dout_0_state
                self.last_di0_toggle_time = now
                self.get_logger().info(
                    f'DI0 rising edge — DO0 → {"ON" if self.tool_dout_0_state else "OFF"}')
                if self.io_connected and self.rtde_io is not None:
                    try:
                        self.rtde_io.setToolDigitalOut(0, self.tool_dout_0_state)
                    except Exception as e:
                        self.get_logger().error(f"Failed to set tool DO0: {e}")
                        self.io_connected = False
            self.tool_din_0_prev = self.tool_digital_in_0

            # DI1 rising edge → toggle gripper
            if (self.tool_digital_in_1 and not self.tool_din_1_prev
                    and (now - self.last_di1_toggle_time) >= self.DEBOUNCE_SECONDS):
                self.gripper_toggle_closed = not self.gripper_toggle_closed
                grip_target = 255 if self.gripper_toggle_closed else 0
                self.last_di1_toggle_time = now
                self.get_logger().info(
                    f"DI1 rising edge — gripper → "
                    f'{"CLOSED" if self.gripper_toggle_closed else "OPEN"}')
                if self.gripper_connected and self.robotiq is not None:
                    try:
                        self.robotiq.move_to(grip_target, speed=255, force=50)
                    except Exception:
                        self.gripper_connected = False
            self.tool_din_1_prev = self.tool_digital_in_1

            # While a policy is driving the destination robot, do NOT publish
            # leader joint/gripper targets onto /mirror/* — the
            # destination_writer would otherwise race between the two
            # publishers. We still keep DI0/DI1 alive so the user can press
            # buttons on the leader pendant.
            if not self.policy_active:
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

            # TCP pose (always published, even when policy_active so the GUI can
            # keep plotting positions).
            try:
                tcp = self.rtde.getActualTCPPose()
                if tcp is not None and len(tcp) >= 6:
                    tp = Float64MultiArray()
                    tp.data = [float(v) for v in tcp[:6]]
                    self.mirror_tcp_pose_pub.publish(tp)
            except Exception:
                pass

        except Exception as e:
            self.get_logger().error(f"RTDE read error: {e}")
            self.rtde_connected = False

    # ── logging ──────────────────────────────────────────────────

    def log_status(self) -> None:
        r = "OK" if self.rtde_connected else "DOWN"
        g = "OK" if self.gripper_connected else "DOWN"
        pos = ", ".join(f"{p:.3f}" for p in self.current_joint_positions)
        # Jerk diagnostics: actual publish rate, max velocity, max delta
        max_vel = max((abs(v) for v in self.current_joint_velocities), default=0.0)
        max_delta = 0.0
        if self.last_pub_positions is not None and self.current_joint_positions:
            max_delta = max(abs(a - b) for a, b in
                            zip(self.current_joint_positions, self.last_pub_positions))
        self.last_pub_positions = list(self.current_joint_positions)
        self.get_logger().debug(
            f"RTDE[{r}] PubHz={self.publish_count} maxV={max_vel:.3f} maxD/s={max_delta:.3f} | "
            f"Joints: [{pos}] | "
            f"Gripper[{g}]: {self.gripper_position_raw}/255 "
            f'({"closed" if self.gripper_is_closed else "open"}) | '
            f'DI0:{"ON" if self.tool_digital_in_0 else "OFF"} '
            f'DI1:{"ON" if self.tool_digital_in_1 else "OFF"} | '
            f'DO0:{"ON" if self.tool_dout_0_state else "OFF"} '
            f'Grip:{"CLOSED" if self.gripper_toggle_closed else "OPEN"}'
        )
        self.publish_count = 0

    # ── cleanup ──────────────────────────────────────────────────

    def destroy_node(self) -> None:
        # Best-effort: leave the source arm in a safe, non-freedrive state.
        if self.rtde_c is not None:
            with contextlib.suppress(Exception):
                if self.teach_mode_active:
                    self.rtde_c.endTeachMode()
                    self.teach_mode_active = False
            with contextlib.suppress(Exception):
                self.rtde_c.stopScript()
            with contextlib.suppress(Exception):
                self.rtde_c.disconnect()
        if self.rtde is not None:
            with contextlib.suppress(Exception):
                self.rtde.disconnect()
        if self.rtde_io is not None:
            with contextlib.suppress(Exception):
                self.rtde_io.disconnect()
        if self.robotiq is not None:
            self.robotiq.disconnect()
        super().destroy_node()

    def _policy_active_cb(self, msg: Bool) -> None:
        new_state = bool(msg.data)
        if new_state != self._policy_log_state:
            self._policy_log_state = new_state
            if new_state:
                self.get_logger().info(
                    "/policy/active=True — suppressing leader → /mirror/* "
                    "publishes; destination_writer will follow the policy.")
            else:
                self.get_logger().info(
                    "/policy/active=False — resuming leader → /mirror/* "
                    "publishes.")
        self.policy_active = new_state

    # ── recording lifecycle → source teachMode ──────────────────

    def _recorder_active_cb(self, msg: Bool) -> None:
        """Engage teachMode while a recording is active; on stop, end teachMode
        and move the source arm back to HOME.

        Rising  edge (False→True): teachMode() so the operator can
                                    freedrive-demonstrate the task.
        Falling edge (True→False): endTeachMode() + moveJ(HOME) in a background
                                    thread so the ROS executor stays responsive.
        """
        new_state = bool(msg.data)
        was_active = self.recorder_active
        self.recorder_active = new_state
        if new_state == was_active:
            return

        if not self.auto_teach_mode:
            return
        if not self.control_connected or self.rtde_c is None:
            # Last-ditch reconnect on the rising edge: pendant may have just
            # been switched from Local to Remote. Try once before giving up so
            # the very next START press already works.
            if new_state and not self._try_connect_control(quiet=False):
                self.get_logger().warn(
                    "Recorder state changed but RTDE Control not connected on "
                    "source — skipping teachMode handling.")
                return
            elif not new_state:
                # Stop path with no Control — nothing to end / move.
                return

        if new_state:
            # START of recording → engage freedrive.
            try:
                ok = self.rtde_c.teachMode()
                self.teach_mode_active = bool(ok)
                self.get_logger().info(
                    f"[recorder] /recorder/active=True → teachMode() "
                    f"returned {ok}")
                if not ok:
                    self.get_logger().warn(
                        "teachMode() refused — controller may be in Local "
                        "mode or protective-stopped.")
            except Exception as e:
                self.teach_mode_active = False
                self.get_logger().error(f"teachMode() raised: {e}")
            return

        # STOP of recording → endTeachMode + moveJ(HOME) in background.
        self.get_logger().info(
            "[recorder] /recorder/active=False → ending teachMode and "
            "returning source to HOME.")
        with self._motion_lock:
            if self._motion_thread is not None and self._motion_thread.is_alive():
                self.get_logger().warn(
                    "Auto-home worker already running; ignoring new request.")
                return
            t = threading.Thread(
                target=self._end_teach_and_go_home,
                name="source-auto-home", daemon=True)
            self._motion_thread = t
            t.start()

    def _end_teach_and_go_home(self) -> None:
        """Runs on a background thread — blocking calls are fine here."""
        try:
            if self.teach_mode_active and self.rtde_c is not None:
                try:
                    self.rtde_c.endTeachMode()
                except Exception as e:
                    self.get_logger().error(f"endTeachMode() raised: {e}")
                self.teach_mode_active = False
                # Brief settle so the controller leaves freedrive cleanly before
                # we issue a motion command.
                time.sleep(0.3)
            if self.rtde_c is None:
                return
            try:
                ok = self.rtde_c.moveJ(
                    list(self.HOME_POSITIONS),
                    self.HOME_VELOCITY,
                    self.HOME_ACCEL,
                )
                self.get_logger().info(
                    f"[recorder] source moveJ(HOME) returned {ok}")
            except Exception as e:
                self.get_logger().error(f"moveJ(HOME) raised: {e}")
        except Exception as e:
            self.get_logger().error(f"auto-home worker crashed: {e}")


def main(args: list[str] | None = None) -> None:
    if not _RTDE_AVAILABLE:
        _LOGGER.error("ur_rtde not installed!  pip install ur_rtde")
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
