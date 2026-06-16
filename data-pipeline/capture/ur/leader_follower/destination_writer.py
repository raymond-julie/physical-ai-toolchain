#!/usr/bin/env python3
"""Destination robot writer for the UR leader/follower teleop+record tool.

State-machine-based destination controller that:

1. ALIGNING  — slowly moves to the recorder home position on startup.
2. IDLE      — holds at home, waits for a DI0 button press.
3. MIRRORING — full-speed mirroring of the source robot.
4. RETURNING — slowly returns to home after DI0 is pressed again.

DI0 toggles between MIRRORING and RETURNING. DI1 toggles the Robotiq gripper
open/closed at any time. A Bool on ``/recorder/active`` tells the recorder node
when to start/stop recording.

Destination robot IP default: 192.168.1.90

Requirements:
    pip install ur_rtde
"""

from __future__ import annotations

import contextlib
import math
import socket
import time
import traceback
from collections.abc import Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    RTDE_AVAILABLE = True
except ImportError:  # pragma: no cover
    RTDE_AVAILABLE = False
    print("WARNING: ur_rtde not installed. Install with: pip install ur_rtde")


WAITING = "WAITING"  # Waiting for GUI confirmation before connecting
ALIGNING = "ALIGNING"  # Moving slowly to home position
IDLE = "IDLE"  # At home, waiting for DI0 press
MIRRORING = "MIRRORING"  # Full-speed mirroring active
RETURNING = "RETURNING"  # Slowly returning to home after stop


def clamp_to_limits(value: float, limits: tuple[float, float]) -> float:
    """Clamp ``value`` to the inclusive ``(lo, hi)`` joint limit."""
    lo, hi = limits
    return max(lo, min(hi, value))


def interpolate_toward(
    current: list[float],
    target: Sequence[float],
    max_step: float,
) -> tuple[float, float]:
    """Advance ``current`` in place toward ``target`` by at most ``max_step`` per joint.

    Returns ``(worst_error, max_excess)``: the largest absolute per-joint error
    before stepping, and the largest error that exceeded ``max_step`` (0.0 when
    every joint was within a single step).
    """
    worst = 0.0
    excess = 0.0
    for i in range(len(current)):
        err = target[i] - current[i]
        worst = max(worst, abs(err))
        if abs(err) > max_step:
            excess = max(excess, abs(err))
            current[i] += max_step if err > 0 else -max_step
        else:
            current[i] = target[i]
    return worst, excess


def at_home(
    servo_positions: Sequence[float] | None,
    home_positions: Sequence[float],
    threshold: float,
) -> bool:
    """True when every servo joint is within ``threshold`` of the home pose."""
    if servo_positions is None:
        return False
    return all(abs(h - s) < threshold for h, s in zip(home_positions, servo_positions))


class RobotiqSocketWriter:
    """Send gripper commands via the Robotiq socket protocol (port 63352)."""

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
            self._send("SET ACT 1")
            time.sleep(0.1)
            self._drain()
            return True
        except OSError:
            self.connected = False
            return False

    def disconnect(self) -> None:
        if self.sock:
            with contextlib.suppress(OSError):
                self.sock.close()
        self.connected = False

    def _send(self, cmd: str) -> str | None:
        if not self.connected or self.sock is None:
            return None
        try:
            self.sock.sendall(f"{cmd}\n".encode())
            time.sleep(0.01)
            return self.sock.recv(1024).decode().strip()
        except OSError:
            self.connected = False
            return None

    def _drain(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.setblocking(False)
            while True:
                try:
                    self.sock.recv(1024)
                except BlockingIOError:
                    break
            self.sock.setblocking(True)
            self.sock.settimeout(self.timeout)
        except OSError:
            pass

    def move_to(self, position: int, speed: int = 255, force: int = 50) -> bool:
        if not self.connected:
            return False
        try:
            self._send(f"SET POS {int(position)}")
            self._send(f"SET SPE {int(speed)}")
            self._send(f"SET FOR {int(force)}")
            self._send("SET GTO 1")
            return True
        except OSError:
            self.connected = False
            return False


class DestinationWriterNode(Node):
    """State-machine destination writer with DI0 toggle control."""

    DESTINATION_ROBOT_IP = "192.168.1.90"

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    JOINT_LIMITS = [
        (-2 * math.pi, 2 * math.pi),
        (-2 * math.pi, 2 * math.pi),
        (-math.pi, math.pi),
        (-2 * math.pi, 2 * math.pi),
        (-2 * math.pi, 2 * math.pi),
        (-2 * math.pi, 2 * math.pi),
    ]

    # Recorder home position captured from a UR5e follower @ 192.168.1.90.
    HOME_POSITIONS = [
        -1.0007379690753382,  # shoulder_pan_joint
        -2.3289038143553675,  # shoulder_lift_joint
        -1.58408784866333,  # elbow_joint
        -0.08030410230670171,  # wrist_1_joint
        -0.9163215796100062,  # wrist_2_joint
        -2.3030503431903284,  # wrist_3_joint
    ]

    def __init__(self) -> None:
        super().__init__("destination_writer")

        self.declare_parameter("robot_ip", self.DESTINATION_ROBOT_IP)
        self.declare_parameter("enable_motion", False)
        self.declare_parameter("servo_time", 0.008)
        self.declare_parameter("servo_lookahead", 0.05)
        self.declare_parameter("servo_gain", 200)
        self.declare_parameter("max_velocity", 1.5)
        self.declare_parameter("max_acceleration", 3.0)
        self.declare_parameter("gripper_speed", 255)
        self.declare_parameter("gripper_force", 50)
        self.declare_parameter("alignment_speed", 0.1)
        self.declare_parameter("alignment_threshold", 0.02)
        self.declare_parameter("use_home", True)
        # Stale-message timeout: if no fresh /mirror/joint_states sample has
        # arrived within this many seconds, freeze the servo target instead of
        # chasing a stale value (avoids the "catch-up snap" when samples resume
        # after a CPU/network stall).
        self.declare_parameter("stale_timeout", 0.2)

        self.robot_ip = self.get_parameter("robot_ip").value
        self.enable_motion = self.get_parameter("enable_motion").value
        self.servo_time = self.get_parameter("servo_time").value
        self.servo_lookahead = self.get_parameter("servo_lookahead").value
        self.servo_gain = self.get_parameter("servo_gain").value
        self.max_velocity = self.get_parameter("max_velocity").value
        self.max_acceleration = self.get_parameter("max_acceleration").value
        self.gripper_speed = self.get_parameter("gripper_speed").value
        self.gripper_force = self.get_parameter("gripper_force").value
        self.alignment_speed = self.get_parameter("alignment_speed").value
        self.alignment_threshold = self.get_parameter("alignment_threshold").value
        self.use_home = self.get_parameter("use_home").value
        self.stale_timeout = self.get_parameter("stale_timeout").value

        # State machine.
        self.state = WAITING  # Wait for GUI confirmation before connecting
        self.motion_confirmed = False
        self.servo_positions: list[float] | None = None  # positions sent to servoJ
        self.source_joint_positions: list[float] | None = None  # latest source positions
        self.mirroring_aligned = False  # catch-up phase complete?
        self.target_gripper_position = 0.0
        self.last_sent_gripper = -1.0
        self.joint_states_received = False
        self.gripper_received = False
        self.commands_sent = 0

        # DI0 toggle tracking (received via /mirror/tool_digital_input_0).
        self.di0_state = False
        self.di0_prev = False
        self.last_di0_toggle_time = 0.0
        self.DEBOUNCE_SECONDS = 0.5

        # Diagnostics for jerk debugging.
        self.msgs_received = 0
        self.last_msg_time = 0.0
        self.max_target_jump = 0.0
        self.max_clamp_excess = 0.0
        self.stale_ticks = 0
        self._prev_target_for_jump: list[float] | None = None

        # Connections.
        self.rtde_control: RTDEControlInterface | None = None
        self.rtde_receive: RTDEReceiveInterface | None = None
        self.robotiq: RobotiqSocketWriter | None = None
        self.rtde_connected = False
        self.gripper_connected = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # RELIABLE shallow-queue QoS for joint mirroring — prevents the silent
        # sample drops that BEST_EFFORT exhibits under load (cameras + bag
        # recording), which were causing the "lag-then-snap" jerk.
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(JointState, "/mirror/joint_states", self._joint_cb, joint_qos)
        self.create_subscription(Float64, "/mirror/gripper/position", self._gripper_cb, qos)
        self.create_subscription(Bool, "/mirror/tool_digital_input_0", self._di0_cb, qos)
        self.create_subscription(Bool, "/gui/toggle", self._gui_toggle_cb, qos)
        self.create_subscription(Bool, "/gui/confirm_motion", self._confirm_motion_cb, qos)
        self.create_subscription(Bool, "/gui/use_home", self._use_home_cb, qos)

        self.recorder_active_pub = self.create_publisher(Bool, "/recorder/active", qos)
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", qos)
        self.state_pub = self.create_publisher(String, "/destination/state", qos)
        self.dest_gripper_pub = self.create_publisher(Float64, "/destination/gripper/position", qos)

        # Do NOT connect to the robot yet — wait for GUI confirmation.
        self.create_timer(0.008, self._servo_tick)  # 125 Hz
        self.create_timer(0.2, self._gripper_tick)  # 5 Hz
        self.create_timer(1.0, self._log_tick)  # 1 Hz
        self.create_timer(3.0, self._reconnect_tick)  # reconnect

        self.get_logger().info(f"Destination Writer started — {self.robot_ip}")
        self.get_logger().info(f'Motion: {"ENABLED" if self.enable_motion else "DISABLED"}')
        self.get_logger().info(
            f'Home position: {"ENABLED" if self.use_home else "DISABLED (direct mirroring)"}'
        )
        if self.use_home:
            self.get_logger().info(f"Home joints: {[round(p, 3) for p in self.HOME_POSITIONS]}")
        if not self.enable_motion:
            self.get_logger().warn("Motion DISABLED — use -p enable_motion:=true")
        if not self.use_home:
            self.get_logger().info("Direct mode — press DI0 to start/stop mirroring")
        self.get_logger().info("⏳ WAITING for motion confirmation from GUI...")

    def connect_rtde(self) -> None:
        if not RTDE_AVAILABLE:
            self.get_logger().error("ur_rtde not installed!")
            return
        try:
            self.get_logger().info(f"Connecting RTDE to {self.robot_ip}...")
            self.rtde_control = RTDEControlInterface(self.robot_ip)
            self.rtde_receive = RTDEReceiveInterface(self.robot_ip)
            self.rtde_connected = True
            self.get_logger().info("RTDE connected!")
        except Exception as exc:
            self.get_logger().error(f"RTDE connection failed: {exc}")
            self.rtde_connected = False

    def connect_gripper(self) -> None:
        try:
            self.get_logger().info(f"Connecting Robotiq gripper at {self.robot_ip}:63352...")
            self.robotiq = RobotiqSocketWriter(self.robot_ip)
            if self.robotiq.connect():
                self.gripper_connected = True
                self.get_logger().info("Robotiq gripper connected!")
            else:
                self.gripper_connected = False
        except Exception as exc:
            self.get_logger().error(f"Gripper connection failed: {exc}")
            self.gripper_connected = False

    def _use_home_cb(self, msg: Bool) -> None:
        """Handle the GUI use_home preference (sent before confirm)."""
        if not self.motion_confirmed:
            self.use_home = msg.data
            self.get_logger().info(f"Use home set to: {self.use_home}")

    def _confirm_motion_cb(self, msg: Bool) -> None:
        """Handle GUI motion confirmation."""
        if msg.data and not self.motion_confirmed:
            self.motion_confirmed = True
            self.get_logger().info("✅ Motion CONFIRMED from GUI — connecting to robot...")
            self.connect_rtde()
            self.connect_gripper()
            self.state = ALIGNING if self.use_home else IDLE
            self.get_logger().info(f"State → {self.state}")

    def _reconnect_tick(self) -> None:
        if not self.motion_confirmed:
            return
        if RTDE_AVAILABLE and not self.rtde_connected:
            self.connect_rtde()
        if not self.gripper_connected:
            self.connect_gripper()

    def _joint_cb(self, msg: JointState) -> None:
        if self.source_joint_positions is None:
            self.source_joint_positions = [0.0] * 6
        try:
            new_target = list(self.source_joint_positions)
            for i, name in enumerate(msg.name):
                if name in self.JOINT_NAMES:
                    idx = self.JOINT_NAMES.index(name)
                    if i < len(msg.position):
                        new_target[idx] = clamp_to_limits(msg.position[i], self.JOINT_LIMITS[idx])
            # Diagnostic: jump in target between consecutive msgs.
            if self._prev_target_for_jump is not None:
                jump = max(abs(a - b) for a, b in zip(new_target, self._prev_target_for_jump))
                self.max_target_jump = max(self.max_target_jump, jump)
            self._prev_target_for_jump = list(new_target)
            self.source_joint_positions = new_target
            self.joint_states_received = True
            self.last_msg_time = time.monotonic()
            self.msgs_received += 1
        except Exception as exc:
            self.get_logger().error(f"Joint callback error: {exc}")

    def _gripper_cb(self, msg: Float64) -> None:
        self.target_gripper_position = max(0.0, min(1.0, msg.data))
        self.gripper_received = True

    def _di0_cb(self, msg: Bool) -> None:
        self.di0_state = msg.data

    def _gui_toggle_cb(self, msg: Bool) -> None:
        """Handle a GUI-triggered start/stop (same as a DI0 press)."""
        if msg.data:
            now = time.monotonic()
            if (now - self.last_di0_toggle_time) >= self.DEBOUNCE_SECONDS:
                self.last_di0_toggle_time = now
                self._handle_di0_toggle()
                self.get_logger().info("GUI toggle received")

    def _interpolate_toward(self, target: Sequence[float], max_speed: float) -> float:
        """Move ``servo_positions`` toward ``target`` at ``max_speed`` rad/s."""
        max_step = max_speed * self.servo_time
        worst, excess = interpolate_toward(self.servo_positions, target, max_step)
        self.max_clamp_excess = max(self.max_clamp_excess, excess)
        return worst

    def _at_home(self) -> bool:
        return at_home(self.servo_positions, self.HOME_POSITIONS, self.alignment_threshold)

    def _servo_tick(self) -> None:
        if not self.motion_confirmed:
            return
        if not self.enable_motion:
            return
        if not self.rtde_connected or self.rtde_control is None:
            return

        # Initialise servo_positions once from the actual robot pose.
        if self.servo_positions is None:
            try:
                self.servo_positions = list(self.rtde_receive.getActualQ())
            except Exception:
                return

        # DI0 rising-edge detection.
        now = time.monotonic()
        if self.di0_state and not self.di0_prev:
            dt = now - self.last_di0_toggle_time
            if dt >= self.DEBOUNCE_SECONDS:
                self.last_di0_toggle_time = now
                self.get_logger().info(f"DI0 rising edge accepted (dt={dt:.3f}s, state={self.state})")
                self._handle_di0_toggle()
            else:
                self.get_logger().warn(f"DI0 rising edge SUPPRESSED by debounce (dt={dt:.3f}s)")
        self.di0_prev = self.di0_state

        if self.state == ALIGNING:
            worst = self._interpolate_toward(self.HOME_POSITIONS, self.alignment_speed)
            if worst < self.alignment_threshold:
                self.state = IDLE
                self.get_logger().info(
                    "✅ ALIGNMENT complete — at home position. "
                    "Press DI0 on source robot to start mirroring."
                )
                self._publish_recorder_active(False)

        elif self.state == IDLE:
            # Hold at home (or hold current position if no home).
            if self.use_home:
                self.servo_positions = list(self.HOME_POSITIONS)

        elif self.state == MIRRORING:
            if self.source_joint_positions is not None and self.joint_states_received:
                # Stale-sample guard: if no fresh mirror msg has arrived within
                # stale_timeout, hold the current servo position rather than
                # chase a stale target. This is the key fix for the "lag then
                # sudden catch-up" symptom seen when cameras + bag recording
                # starve the source-reader callback for tens of milliseconds.
                now = time.monotonic()
                stale = self.last_msg_time > 0.0 and (now - self.last_msg_time) > self.stale_timeout
                if stale:
                    self.stale_ticks += 1
                elif not self.mirroring_aligned:
                    # Catch-up phase: slowly interpolate toward source position.
                    worst = self._interpolate_toward(self.source_joint_positions, self.alignment_speed)
                    if worst < self.alignment_threshold:
                        self.mirroring_aligned = True
                        self.get_logger().info("✅ Catch-up complete — full-speed mirroring active")
                else:
                    # Full-speed tracking, rate-limited per servo tick so a
                    # missed source-reader tick does not produce a velocity-
                    # limit lurch on the robot. With normal sub-tick deltas the
                    # interpolator is a no-op passthrough.
                    self._interpolate_toward(self.source_joint_positions, self.max_velocity)

        elif self.state == RETURNING:
            worst = self._interpolate_toward(self.HOME_POSITIONS, self.alignment_speed)
            if worst < self.alignment_threshold:
                self.state = IDLE
                self.get_logger().info(
                    "✅ RETURNED to home — ready for next recording cycle. Press DI0 to start."
                )
                self._publish_recorder_active(False)

        try:
            self.rtde_control.servoJ(
                self.servo_positions,
                self.max_velocity,
                self.max_acceleration,
                self.servo_time,
                self.servo_lookahead,
                self.servo_gain,
            )
            self.commands_sent += 1

            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(self.JOINT_NAMES)
            js.position = list(self.servo_positions)
            self.joint_state_pub.publish(js)
        except Exception as exc:
            self.get_logger().error(f"servoJ failed: {exc}")
            self.rtde_connected = False

    def _handle_di0_toggle(self) -> None:
        if self.state == IDLE:
            # Start mirroring + recording (catch-up phase first).
            self.mirroring_aligned = False
            self.state = MIRRORING
            self._publish_recorder_active(True)
            self.get_logger().info("▶ DI0 pressed — catching up to source position...")

        elif self.state == MIRRORING:
            if self.use_home:
                self.state = RETURNING
                self._publish_recorder_active(False)
                self.get_logger().info(
                    "⏹ DI0 pressed — MIRRORING stopped, returning to home. Recording OFF."
                )
            else:
                self.state = IDLE
                self._publish_recorder_active(False)
                self.get_logger().info(
                    "⏹ DI0 pressed — MIRRORING stopped, holding position. "
                    "Recording OFF. Press DI0 to resume."
                )

        elif self.state == WAITING:
            self.get_logger().warn("DI0 pressed before motion confirmed — ignoring")

        elif self.state == ALIGNING:
            self.get_logger().warn("DI0 pressed during alignment — ignoring")

        elif self.state == RETURNING:
            self.get_logger().warn("DI0 pressed while returning — ignoring")

    def _publish_recorder_active(self, active: bool) -> None:
        msg = Bool()
        msg.data = active
        self.recorder_active_pub.publish(msg)
        # Diagnostic: log every transition with caller state so unexpected
        # episode splits can be explained.
        caller = traceback.extract_stack()[-2]
        self.get_logger().info(
            f"/recorder/active <- {active}  (state={self.state}  "
            f"caller={caller.name}:{caller.lineno})"
        )

    def _gripper_tick(self) -> None:
        if not self.motion_confirmed:
            return
        if not self.enable_motion:
            return
        if not self.gripper_received:
            return

        # Always publish destination gripper position for the GUI.
        grip_msg = Float64()
        grip_msg.data = self.target_gripper_position
        self.dest_gripper_pub.publish(grip_msg)

        if not self.gripper_connected or self.robotiq is None:
            return
        if abs(self.target_gripper_position - self.last_sent_gripper) < 0.05:
            return
        self.last_sent_gripper = self.target_gripper_position
        grip_int = max(0, min(255, int(self.target_gripper_position * 255)))
        if not self.robotiq.move_to(grip_int, self.gripper_speed, self.gripper_force):
            self.gripper_connected = False

    def _log_tick(self) -> None:
        m = "ON" if self.enable_motion else "OFF"
        r = "OK" if self.rtde_connected else "DOWN"
        g = "OK" if self.gripper_connected else "DOWN"

        pos = ", ".join(f"{p:.3f}" for p in self.servo_positions) if self.servo_positions else "N/A"

        extra = ""
        if self.state in (ALIGNING, RETURNING) and self.servo_positions:
            worst = max(abs(h - s) for h, s in zip(self.HOME_POSITIONS, self.servo_positions))
            deg = math.degrees(worst)
            extra = f" | err={worst:.3f}rad ({deg:.1f}°)"

        self.get_logger().info(
            f"[{self.state}] Motion[{m}] RTDE[{r}] Grip[{g}] | "
            f"Joints: [{pos}] | "
            f"Gripper: {self.target_gripper_position:.3f} | "
            f"Cmds/s: {self.commands_sent} MsgsRx/s: {self.msgs_received} "
            f"maxJump={self.max_target_jump:.3f} clampExc={self.max_clamp_excess:.3f} "
            f"staleTicks={self.stale_ticks}{extra}"
        )
        self.commands_sent = 0
        self.msgs_received = 0
        self.max_target_jump = 0.0
        self.max_clamp_excess = 0.0
        self.stale_ticks = 0

        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

    def destroy_node(self) -> bool:
        with contextlib.suppress(Exception):
            self._publish_recorder_active(False)
        if self.rtde_control is not None:
            with contextlib.suppress(Exception):
                self.rtde_control.servoStop()
                self.rtde_control.stopScript()
                self.rtde_control.disconnect()
        if self.rtde_receive is not None:
            with contextlib.suppress(Exception):
                self.rtde_receive.disconnect()
        if self.robotiq is not None:
            self.robotiq.disconnect()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    if not RTDE_AVAILABLE:
        print("ERROR: ur_rtde not installed!  pip install ur_rtde")
        return
    rclpy.init(args=args)
    node = DestinationWriterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().info("Shutting down Destination Writer")
    finally:
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
