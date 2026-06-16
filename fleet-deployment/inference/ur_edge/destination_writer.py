#!/usr/bin/env python3
"""Destination robot writer for the UR edge runtime.

State-machine-based destination controller that:

1. ALIGNING  — slowly moves to the recorder home position on startup
2. IDLE      — holds at home, waits for DI0 button press
3. MIRRORING — full-speed mirroring of the source robot
4. RETURNING — slowly returns to home after DI0 is pressed again

DI0 toggles between MIRRORING and RETURNING. DI1 toggles the Robotiq gripper
open/closed at any time.

The node publishes a Bool on ``/recorder/active`` so the recorder node knows
when to start/stop recording.

Destination robot IP: 192.168.1.102

Requirements::

    pip install ur_rtde
"""

from __future__ import annotations

import contextlib
import logging
import math
import socket
import time
import traceback

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

_LOGGER = logging.getLogger(__name__)

try:
    # The teach-pendant speed slider is exposed only via RTDEIOInterface in the
    # Python bindings (not RTDEControlInterface). DashboardClient exposes
    # safety-popup / protective-stop unlock, the only way to recover from a
    # collision without a human pressing the pendant.
    from dashboard_client import DashboardClient
    from rtde_control import RTDEControlInterface
    from rtde_io import RTDEIOInterface
    from rtde_receive import RTDEReceiveInterface
    _RTDE_AVAILABLE = True
except ImportError:
    _RTDE_AVAILABLE = False
    _LOGGER.warning("ur_rtde not installed. Install with: pip install ur_rtde")


# ── States ───────────────────────────────────────────────────────────────────

WAITING = "WAITING"        # Waiting for GUI confirmation before connecting
ALIGNING = "ALIGNING"      # Moving slowly to home position
IDLE = "IDLE"              # At home, waiting for DI0 press
MIRRORING = "MIRRORING"    # Full-speed mirroring active
RETURNING = "RETURNING"    # Slowly returning to home after stop
RECOVERING = "RECOVERING"  # Protective stop detected; auto-unlock + home

# UR safety standard requires at least 5 s between protective stop trigger and
# unlock. We add a small margin.
PROTECTIVE_STOP_UNLOCK_DELAY = 5.5


# ── Pure helpers (unit-testable; no hardware/ROS) ────────────────────────────

def clamp_speed_scale(value: float) -> float:
    """Clamp the MIRRORING speed scale to the controller-safe range [0.01, 1.0]."""
    return max(0.01, min(1.0, float(value)))


def clamp_interp_scale(value: float) -> float:
    """Clamp the TCP interpolation horizon multiplier to [0.0, 10.0]."""
    return max(0.0, min(10.0, float(value)))


def di0_next_state(state: str, use_home: bool) -> str | None:
    """Pure DI0-toggle transition.

    Returns the new state, or ``None`` when the press is ignored (state
    unchanged). IDLE starts mirroring; MIRRORING stops to RETURNING (when a home
    pose is used) or IDLE (direct mode). WAITING/ALIGNING/RETURNING ignore DI0.
    """
    if state == IDLE:
        return MIRRORING
    if state == MIRRORING:
        return RETURNING if use_home else IDLE
    return None


def policy_next_state(state: str, policy_active: bool, policy_induced: bool,
                      use_home: bool) -> str | None:
    """Pure ``/policy/active`` transition.

    Returns the new state, or ``None`` when no transition applies. A running
    policy drives IDLE → MIRRORING; when the policy ends, a *policy-induced*
    MIRRORING reverts to RETURNING (home) or IDLE (direct). DI0/GUI-induced
    mirroring is left untouched.
    """
    if policy_active and state == IDLE:
        return MIRRORING
    if (not policy_active) and state == MIRRORING and policy_induced:
        return RETURNING if use_home else IDLE
    return None


def gripper_hysteresis(raw: float, is_closed: bool, last_flip_t: float,
                       now: float, close_threshold: float,
                       open_threshold: float,
                       min_dwell: float) -> tuple[bool, bool]:
    """Binary-gripper hysteresis with dwell. Returns ``(new_is_closed, flipped)``.

    Treats the gripper as binary with asymmetric thresholds plus a minimum dwell
    time between flips. A single noisy action in a policy action chunk (e.g. a
    0.4 between two 0.9s) would otherwise drive a brief partial-open that drops a
    held part. Teach-mode mirroring is unaffected because the operator's gripper
    crosses the full 0↔1 range cleanly.
    """
    desired_closed = is_closed
    if not is_closed and raw >= close_threshold:
        desired_closed = True
    elif is_closed and raw <= open_threshold:
        desired_closed = False
    if desired_closed != is_closed:
        if (now - last_flip_t) < min_dwell:
            # Suppress this flip — too soon after the previous one.
            return is_closed, False
        return desired_closed, True
    return is_closed, False


# ── Robotiq gripper writer ───────────────────────────────────────────────────

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
        with contextlib.suppress(OSError):
            self.sock.setblocking(False)
            while True:
                try:
                    self.sock.recv(1024)
                except BlockingIOError:
                    break
            self.sock.setblocking(True)
            self.sock.settimeout(self.timeout)

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


# ── ROS2 node ────────────────────────────────────────────────────────────────

class DestinationWriterNode(Node):
    """State-machine destination writer with DI0 toggle control."""

    DESTINATION_ROBOT_IP = "192.168.1.102"

    JOINT_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]

    JOINT_LIMITS = [
        (-2 * math.pi, 2 * math.pi),
        (-2 * math.pi, 2 * math.pi),
        (-math.pi, math.pi),
        (-2 * math.pi, 2 * math.pi),
        (-2 * math.pi, 2 * math.pi),
        (-2 * math.pi, 2 * math.pi),
    ]

    # Recorder home position (must match auto_recorder target).
    HOME_POSITIONS = [
        1.3399624824523926,    # shoulder_pan_joint
        -1.2604854863933106,   # shoulder_lift_joint
        1.8152335325824183,    # elbow_joint
        -2.3439699612059535,   # wrist_1_joint
        -1.5236032644854944,   # wrist_2_joint
        -0.24175817171205694,  # wrist_3_joint
    ]

    def __init__(self) -> None:
        super().__init__("destination_writer")

        # ── Parameters ──
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
        # ── Gripper hysteresis (anti-flicker for policy rollouts) ────────
        #   - commanded ≥ close_threshold  -> CLOSE  (sends POS=255)
        #   - commanded ≤ open_threshold   -> OPEN   (sends POS=0)
        #   - else: no change
        # Min dwell suppresses opposite-direction commands within the window
        # after a flip. See :func:`gripper_hysteresis`.
        self.declare_parameter("gripper_close_threshold", 0.55)
        self.declare_parameter("gripper_open_threshold", 0.30)
        self.declare_parameter("gripper_min_dwell", 0.7)
        # Speed scaling factor applied to MIRRORING motion (servoJ/servoL
        # velocity + acceleration and the per-tick interpolator step). Default
        # 10% so policy rollouts (ACT/SmolVLA/GR00T) move slowly on first run;
        # the GUI publishes /gui/speed_scale to change it live. Range clamped to
        # [0.01, 1.0]. Alignment / home returns are unaffected (they use
        # alignment_speed).
        self.declare_parameter("speed_scale", 0.10)
        # TCP interpolation horizon multiplier. 0 disables the interpolator (raw
        # setpoints sent every tick; useful for observing the model's native
        # jerk). 1.0 = use the measured inter-arrival as the ramp horizon. >1 =
        # lazier ramp; <1 = snappier.
        self.declare_parameter("interp_scale", 1.0)
        # Stale-message timeout: if no fresh /mirror/joint_states sample has
        # arrived within this many seconds, freeze the servo target instead of
        # continuing to chase a stale value (avoids the "catch-up snap" when
        # samples resume after a CPU/network stall). Default raised to 1.5s so
        # closed-loop SmolVLA inference (~700ms per chunk on Jetson Orin) does
        # not constantly trip this guard.
        self.declare_parameter("stale_timeout", 1.5)

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
        self.gripper_close_threshold = float(self.get_parameter("gripper_close_threshold").value)
        self.gripper_open_threshold = float(self.get_parameter("gripper_open_threshold").value)
        self.gripper_min_dwell = float(self.get_parameter("gripper_min_dwell").value)
        self.speed_scale = clamp_speed_scale(self.get_parameter("speed_scale").value)
        self.interp_scale = clamp_interp_scale(self.get_parameter("interp_scale").value)
        self.stale_timeout = self.get_parameter("stale_timeout").value

        # Tracks the last value pushed to the UR controller via setSpeedSlider so
        # we only call it on actual change. Initial value is intentionally
        # out-of-range so the first apply always fires once we're connected.
        self._applied_slider: float = -1.0

        # ── State machine ──
        self.state = WAITING  # Wait for GUI confirmation before connecting
        self.motion_confirmed = False
        self.servo_positions: list[float] | None = None        # positions sent to servoJ
        self.source_joint_positions: list[float] | None = None  # latest source positions
        self.mirroring_aligned = False       # catch-up phase complete?
        self.target_gripper_position = 0.0
        self.last_sent_gripper = -1.0
        # Binary state used by the hysteresis layer in _gripper_tick.
        self._gripper_is_closed = False
        self._gripper_last_flip_t = 0.0
        self.joint_states_received = False
        self.gripper_received = False
        self.commands_sent = 0

        # DI0 toggle tracking (received via /mirror/tool_digital_input_0)
        self.di0_state = False
        self.di0_prev = False
        self.last_di0_toggle_time = 0.0
        self.DEBOUNCE_SECONDS = 0.5

        # Diagnostics for jerk debugging
        self.msgs_received = 0
        self.last_msg_time = 0.0
        self.max_target_jump = 0.0
        self.max_clamp_excess = 0.0
        self.stale_ticks = 0
        self._prev_target_for_jump: list[float] | None = None

        # ── Cartesian (TCP) target ──
        # Optional Cartesian command path. When a fresh TCP target is available
        # (received within ``stale_timeout``) and the state machine is
        # MIRRORING, the servo tick switches from ``servoJ`` to ``servoL`` —
        # i.e. the UR controller's own firmware does the IK from base-frame pose
        # to joint targets. Falls back to joint mirroring transparently when TCP
        # samples go stale. Pose convention is the standard URScript
        # ``[x, y, z, rx, ry, rz]`` (axis-angle, metres / radians).
        self.target_tcp_pose: list[float] | None = None
        self.last_tcp_msg_time = 0.0
        self.tcp_msgs_received = 0
        # Inter-setpoint interpolation state. GR00T TCP-EE rollouts at 10% speed
        # publish setpoints ~670 ms apart, so the bare `servoL(target, ...)`
        # reaches the target in ~100 ms then holds for the remaining ~570 ms,
        # producing a discrete step-step-step motion. We instead linearly
        # interpolate the pose between the previous and current target across the
        # measured inter-arrival interval, so every 8 ms servo tick gets a fresh
        # in-between pose and motion looks continuous.
        self._tcp_prev_target: list[float] | None = None  # previous absolute setpoint
        self._tcp_target_arrival = 0.0       # monotonic time current target arrived
        self._tcp_inter_arrival_ema = 0.1    # EMA of inter-arrival interval [s]

        # Connections
        self.rtde_control = None
        self.rtde_receive = None
        self.rtde_io = None
        self.dashboard = None
        self.robotiq: RobotiqSocketWriter | None = None
        self.rtde_connected = False
        self.gripper_connected = False
        self.dashboard_connected = False

        # Protective-stop recovery bookkeeping.
        self._protective_stop_t: float = 0.0   # monotonic time we detected
        self._last_unlock_attempt_t: float = 0.0
        # Pre-stop state so we know whether to silently resume or just go IDLE.
        self._pre_stop_state: str | None = None
        # Operator ack — recovery WILL NOT auto-run until the GUI publishes True
        # on /gui/recovery_ack. This is a safety interlock: after a collision the
        # workspace may be unsafe (dropped part, person at arm, fixture shifted)
        # and the arm must not silently home itself without a human confirming it
        # is clear.
        self._recovery_ack: bool = False

        # ── QoS ──
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        # RELIABLE shallow-queue QoS for joint mirroring — prevents the silent
        # sample drops that BEST_EFFORT exhibits under load (cameras + bag
        # recording), which were causing the "lag-then-snap" jerk.
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        # ── Subscribers ──
        self.create_subscription(JointState, "/mirror/joint_states",
                                 self._joint_cb, joint_qos)
        self.create_subscription(Float64MultiArray, "/mirror/tcp_pose_cmd",
                                 self._tcp_pose_cb, joint_qos)
        self.create_subscription(Float64, "/mirror/gripper/position",
                                 self._gripper_cb, qos)
        self.create_subscription(Bool, "/mirror/tool_digital_input_0",
                                 self._di0_cb, qos)
        self.create_subscription(Bool, "/gui/toggle",
                                 self._gui_toggle_cb, qos)
        self.create_subscription(Bool, "/gui/confirm_motion",
                                 self._confirm_motion_cb, qos)
        self.create_subscription(Bool, "/gui/use_home",
                                 self._use_home_cb, qos)
        self.create_subscription(Float64, "/gui/speed_scale",
                                 self._speed_scale_cb, qos)
        self.create_subscription(Float64, "/gui/interp_scale",
                                 self._interp_scale_cb, qos)
        # Operator must explicitly acknowledge a protective-stop recovery before
        # the arm is allowed to unlock and home itself.
        self.create_subscription(Bool, "/gui/recovery_ack",
                                 self._recovery_ack_cb, qos)
        # When the AI policy is driving the robot we don't want to record
        # demonstrations. Listen to /policy/active so the IDLE → MIRRORING
        # transition can suppress /recorder/active.
        latch_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.policy_active = False
        # Tracks whether the current MIRRORING state was started by the policy
        # (so we can revert to IDLE when the policy stops). DI0/GUI toggles clear
        # this flag — those should NOT be undone by /policy/active going False.
        self._policy_induced_mirror = False
        self.create_subscription(
            Bool, "/policy/active", self._policy_active_cb, latch_qos)

        # ── Publishers ──
        self.recorder_active_pub = self.create_publisher(Bool, "/recorder/active", qos)
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", qos)

        # State publisher (for GUI)
        self.state_pub = self.create_publisher(String, "/destination/state", qos)

        # Destination gripper position publisher (for GUI)
        self.dest_gripper_pub = self.create_publisher(Float64, "/destination/gripper/position", qos)

        # ── True destination-side measured-state publishers ──
        # Distinct from the existing ``/joint_states`` (which carries the
        # *commanded* servo target this writer is currently sending). These
        # three topics carry what the destination robot has actually *achieved*
        # and are the canonical ``observation.*`` signals for the LeRobot
        # dataset. Published from a dedicated 30 Hz timer reading ``getActualQ``
        # / ``getActualTCPPose``.
        self.dest_joint_states_pub = self.create_publisher(
            JointState, "/destination/joint_states", qos)
        self.dest_tcp_pose_pub = self.create_publisher(
            Float64MultiArray, "/destination/tcp_pose", qos)
        # Binary "is the gripper closed" derived from the latest target (>= 0.5
        # means closed). This is a proxy until we wire a true Robotiq POS
        # readback; on the UR10e + 2F-85 the actuation lag is small enough
        # (~150 ms at 255 speed) that the proxy matches ground truth for >95% of
        # frames at 15 Hz.
        self.dest_gripper_closed_pub = self.create_publisher(
            Bool, "/destination/gripper/is_closed", qos)

        # ── Do NOT connect to robot yet — wait for GUI confirmation ──

        # ── Timers ──
        self.create_timer(0.008, self._servo_tick)        # 125 Hz
        self.create_timer(0.2, self._gripper_tick)        # 5 Hz
        # 30 Hz measured-state publisher (dest joints + TCP + gripper bit).
        self.create_timer(1.0 / 30.0, self._dest_state_tick)
        self.create_timer(1.0, self._log_tick)            # 1 Hz
        self.create_timer(3.0, self._reconnect_tick)      # reconnect

        self.get_logger().info(f"Destination Writer started — {self.robot_ip}")
        self.get_logger().info(f'Motion: {"ENABLED" if self.enable_motion else "DISABLED"}')
        self.get_logger().info(
            f'Home position: {"ENABLED" if self.use_home else "DISABLED (direct mirroring)"}')
        if self.use_home:
            self.get_logger().info(f"Home joints: {[round(p, 3) for p in self.HOME_POSITIONS]}")

        if not self.enable_motion:
            self.get_logger().warn("Motion DISABLED — use -p enable_motion:=true")
        if not self.use_home:
            self.get_logger().info("Direct mode — press DI0 to start/stop mirroring")

        self.get_logger().info("⏳ WAITING for motion confirmation from GUI...")

    # ── Connections ──────────────────────────────────────────────

    def connect_rtde(self) -> None:
        if not _RTDE_AVAILABLE:
            self.get_logger().error("ur_rtde not installed!")
            return
        try:
            self.get_logger().info(f"Connecting RTDE to {self.robot_ip}...")
            # Use upper register range (24–47) so we don't collide with
            # EtherNet/IP, PROFINET, or Modbus units which claim the lower range
            # (0–23) on the UR controller.
            flags = (RTDEControlInterface.FLAG_UPLOAD_SCRIPT
                     | RTDEControlInterface.FLAG_UPPER_RANGE_REGISTERS)
            self.rtde_control = RTDEControlInterface(self.robot_ip, flags=flags)
            self.rtde_receive = RTDEReceiveInterface(self.robot_ip)
            # Dedicated IO interface for the teach-pendant speed slider. Failure
            # here is non-fatal — the rest of the system still works, just
            # without firmware-level speed capping.
            try:
                self.rtde_io = RTDEIOInterface(self.robot_ip)
            except Exception as e:
                self.rtde_io = None
                self.get_logger().warn(
                    f"RTDEIOInterface unavailable — speed slider disabled: {e}")
            self.rtde_connected = True
            # Force first apply on next tick.
            self._applied_slider = -1.0
            self.get_logger().info("RTDE connected!")
        except Exception as e:
            self.get_logger().error(f"RTDE connection failed: {e}")
            self.rtde_connected = False
        # Dashboard connection is required to recover from a protective stop.
        # Failure is non-fatal — without it, collisions still need a human at the
        # pendant.
        try:
            self.dashboard = DashboardClient(self.robot_ip)
            self.dashboard.connect()
            self.dashboard_connected = self.dashboard.isConnected()
            if self.dashboard_connected:
                self.get_logger().info("Dashboard connected (protective-stop recovery enabled).")
        except Exception as e:
            self.dashboard = None
            self.dashboard_connected = False
            self.get_logger().warn(
                f"DashboardClient unavailable — auto-recovery from "
                f"collisions disabled: {e}")

    def _desired_slider(self) -> float:
        """Slider value the UR controller should be running at right now.

        Master cap: every phase obeys the GUI ``speed_scale``. The alignment /
        return-to-home interpolation pre-compensates by dividing by this value so
        its effective rad/s stays at ``alignment_speed`` regardless of the slider
        setting.
        """
        return self.speed_scale

    def _apply_speed_slider(self) -> None:
        """Push the desired slider value to the UR controller. No-op when
        unchanged or when not connected. The controller enforces this cap on
        every motion primitive (servoJ/servoL/movej/...) — it is the same
        fraction the teach-pendant slider drives.

        Note: in remote control mode the polyscope slider widget is greyed out
        and may not visually reflect the value, but the underlying
        ``target_speed_fraction`` register *does* update.
        ``rtde_receive.getSpeedScaling()`` is the ground truth.
        """
        if not self.rtde_connected or getattr(self, "rtde_io", None) is None:
            return
        want = self._desired_slider()
        if abs(want - self._applied_slider) < 1e-4:
            return
        try:
            ok = self.rtde_io.setSpeedSlider(want)
        except Exception as e:
            self.get_logger().warn(f"setSpeedSlider({want:.2f}) failed: {e}")
            return
        if not ok:
            self.get_logger().warn(
                f"setSpeedSlider({want:.2f}) returned False "
                "(controller refused — check remote control mode is ON)")
            return
        self._applied_slider = want
        # Read back the slider position the controller has accepted. NOTE: use
        # getTargetSpeedFraction() — that is the slider value. getSpeedScaling()
        # returns the runtime trajectory-limiter scaling which stays at 1.0 for
        # servoJ/servoL (they bypass the planner), so it is misleading here.
        readback = None
        try:
            readback = self.rtde_receive.getTargetSpeedFraction()
        except Exception:
            pass
        rb_str = f", readback={readback * 100:.0f}%" if readback is not None else ""
        self.get_logger().info(
            f"UR speed slider -> {want * 100:.0f}% (state={self.state}{rb_str})")

    def connect_gripper(self) -> None:
        try:
            self.get_logger().info(f"Connecting Robotiq gripper at {self.robot_ip}:63352...")
            self.robotiq = RobotiqSocketWriter(self.robot_ip)
            if self.robotiq.connect():
                self.gripper_connected = True
                self.get_logger().info("Robotiq gripper connected!")
            else:
                self.gripper_connected = False
        except Exception as e:
            self.get_logger().error(f"Gripper connection failed: {e}")
            self.gripper_connected = False

    def _use_home_cb(self, msg: Bool) -> None:
        """Handle GUI use_home preference (sent before confirm)."""
        if not self.motion_confirmed:
            self.use_home = msg.data

    def _speed_scale_cb(self, msg: Float64) -> None:
        new_scale = clamp_speed_scale(msg.data)
        if abs(new_scale - self.speed_scale) > 1e-4:
            self.get_logger().info(
                f"Speed scale: {self.speed_scale * 100:.0f}% -> {new_scale * 100:.0f}%")
            self.speed_scale = new_scale
            # If we're currently mirroring, push the new cap to the controller
            # immediately rather than waiting for the next servo tick (which
            # would only matter for the log line — _apply_speed_slider is
            # idempotent and called every tick).
            self._apply_speed_slider()

    def _interp_scale_cb(self, msg: Float64) -> None:
        new_scale = clamp_interp_scale(msg.data)
        if abs(new_scale - self.interp_scale) > 1e-4:
            label = "OFF (snap)" if new_scale <= 0.0 else f"{new_scale:.2f}x"
            self.get_logger().info(
                f"Interp scale: {self.interp_scale:.2f} -> {label}")
            self.interp_scale = new_scale

    def _policy_active_cb(self, msg: Bool) -> None:
        new_state = bool(msg.data)
        if new_state != self.policy_active:
            self.get_logger().info(
                f"/policy/active <- {new_state} "
                "(recording behaviour: always honours START/STOP regardless "
                "of policy state)")
        self.policy_active = new_state

        # Auto-drive the state machine so a running policy actually moves the
        # arm. Without this the arm stays in IDLE and only the gripper tracks
        # /mirror/* (gripper is handled outside the state machine).
        if not self.motion_confirmed:
            return
        nxt = policy_next_state(self.state, new_state,
                                self._policy_induced_mirror, self.use_home)
        if nxt is None:
            return
        if new_state:
            self.mirroring_aligned = False
            self.state = nxt  # MIRRORING
            self._policy_induced_mirror = True
            self.get_logger().info(
                "▶ /policy/active=True — entering MIRRORING (policy-driven). "
                "Recording NOT auto-started; press DI0/GUI START to record.")
        else:
            self._policy_induced_mirror = False
            self.state = nxt  # RETURNING or IDLE
            if self.use_home:
                self.get_logger().info(
                    "⏹ /policy/active=False — returning to home (policy ended).")
            else:
                self.get_logger().info(
                    "⏹ /policy/active=False — holding position (policy ended).")

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
            return  # Don't try to connect until confirmed
        if _RTDE_AVAILABLE and not self.rtde_connected:
            self.connect_rtde()
        if not self.gripper_connected:
            self.connect_gripper()
        # Drive protective-stop recovery on this same 3 s cadence so we don't
        # hammer the dashboard socket.
        if self.state == RECOVERING:
            self._try_recover_from_protective_stop()

    # ── Protective-stop recovery ─────────────────────────────────

    def _check_protective_stop(self) -> bool:
        """Poll the controller for a latched protective stop.

        On the rising edge, transitions to RECOVERING. Returns True if the robot
        is currently stopped (caller should skip the tick).
        """
        if self.rtde_receive is None:
            return False
        try:
            stopped = bool(self.rtde_receive.isProtectiveStopped())
        except Exception:
            # If receive itself is dead the connection is gone — let the
            # reconnect tick handle it; treat as not-stopped here so we don't
            # loop into RECOVERING on a transient socket.
            return False
        if stopped and self.state != RECOVERING:
            self._enter_recovering("isProtectiveStopped=True")
        return stopped

    def _enter_recovering(self, reason: str) -> None:
        """Transition into RECOVERING from any other state.

        Stops the servo stream, remembers what we were doing, and starts the
        unlock clock. Idempotent.
        """
        if self.state == RECOVERING:
            return
        self._pre_stop_state = self.state
        self.state = RECOVERING
        self._protective_stop_t = time.monotonic()
        self._last_unlock_attempt_t = 0.0
        # Clear any stale ack from a previous incident so the operator is forced
        # to acknowledge THIS one.
        self._recovery_ack = False
        # Stop publishing the recorder-active flag so any concurrent bag
        # recording closes cleanly.
        with contextlib.suppress(Exception):
            self._publish_recorder_active(False)
        # Reset interpolation state — when we resume we want to start fresh from
        # the actual joint pose, not from a stale target.
        self.servo_positions = None
        self.mirroring_aligned = False
        self.get_logger().error(
            f"🛑 Protective stop detected ({reason}). Entering RECOVERING. "
            f"Workspace check required — publish True on /gui/recovery_ack "
            f'(GUI "Recover" button) to unlock and home. '
            f"Previous state: {self._pre_stop_state}.")

    def _recovery_ack_cb(self, msg: Bool) -> None:
        """Operator-initiated recovery confirmation.

        Only meaningful while in RECOVERING; ignored otherwise. A True edge
        unlatches the interlock and lets ``_try_recover_from_protective_stop``
        proceed once the safety cooldown has elapsed.
        """
        if not bool(msg.data):
            return
        if self.state != RECOVERING:
            self.get_logger().info(
                "/gui/recovery_ack received but state != RECOVERING; ignoring.")
            return
        if self._recovery_ack:
            return  # already acked
        self._recovery_ack = True
        self.get_logger().info(
            "✅ Recovery acknowledged by operator. Unlock will proceed "
            "on next reconnect tick (after safety cooldown).")

    def _try_recover_from_protective_stop(self) -> None:
        """Called periodically while in RECOVERING.

        Closes the safety popup, calls ``unlockProtectiveStop`` once the
        mandatory cooldown has elapsed AND the operator has acknowledged, then
        reconnects RTDE control and hands off to ALIGNING so the arm returns
        home.
        """
        # Hard interlock: do nothing until the operator has confirmed the
        # workspace is safe.
        if not self._recovery_ack:
            return
        now = time.monotonic()
        # Throttle attempts to once every 2 s so we don't spam.
        if now - self._last_unlock_attempt_t < 2.0:
            return
        self._last_unlock_attempt_t = now
        # Enforce the UR safety cooldown.
        wait_left = PROTECTIVE_STOP_UNLOCK_DELAY - (now - self._protective_stop_t)
        if wait_left > 0:
            self.get_logger().info(
                f"RECOVERING: waiting {wait_left:.1f}s before unlock attempt.")
            return
        if self.dashboard is None or not self.dashboard_connected:
            try:
                self.dashboard = DashboardClient(self.robot_ip)
                self.dashboard.connect()
                self.dashboard_connected = self.dashboard.isConnected()
            except Exception as e:
                self.get_logger().warn(f"Dashboard reconnect failed: {e}")
                self.dashboard_connected = False
                return
        try:
            self.dashboard.closeSafetyPopup()
        except Exception as e:
            self.get_logger().warn(f"closeSafetyPopup failed: {e}")
        try:
            self.dashboard.unlockProtectiveStop()
        except Exception as e:
            self.get_logger().warn(f"unlockProtectiveStop failed: {e}")
            return
        # Verify the stop cleared. RTDE receive can answer this even when control
        # is disconnected.
        still_stopped = True
        try:
            if self.rtde_receive is not None:
                still_stopped = bool(self.rtde_receive.isProtectiveStopped())
        except Exception:
            pass
        if still_stopped:
            self.get_logger().warn(
                "Protective stop still active after unlock; will retry.")
            return
        # Rebuild RTDE control — the script upload was killed by the stop.
        self.rtde_connected = False
        with contextlib.suppress(Exception):
            if self.rtde_control is not None:
                self.rtde_control.disconnect()
        self.rtde_control = None
        self.connect_rtde()
        if not self.rtde_connected:
            self.get_logger().warn("RTDE reconnect failed; will retry.")
            return
        # Success — hand off to ALIGNING so the arm walks back to home at the
        # conservative alignment speed.
        self.get_logger().info(
            "✅ Protective stop cleared. Returning to home via ALIGNING.")
        self.state = ALIGNING
        self._pre_stop_state = None
        self._recovery_ack = False

    # ── Callbacks ────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState) -> None:
        if self.source_joint_positions is None:
            self.source_joint_positions = [0.0] * 6
        try:
            new_target = list(self.source_joint_positions)
            for i, name in enumerate(msg.name):
                if name in self.JOINT_NAMES:
                    idx = self.JOINT_NAMES.index(name)
                    if i < len(msg.position):
                        lo, hi = self.JOINT_LIMITS[idx]
                        new_target[idx] = max(lo, min(hi, msg.position[i]))
            # Diagnostic: jump in target between consecutive msgs
            if self._prev_target_for_jump is not None:
                jump = max(abs(a - b) for a, b in
                           zip(new_target, self._prev_target_for_jump))
                if jump > self.max_target_jump:
                    self.max_target_jump = jump
            self._prev_target_for_jump = list(new_target)
            self.source_joint_positions = new_target
            self.joint_states_received = True
            self.last_msg_time = time.monotonic()
            self.msgs_received += 1
        except Exception as e:
            self.get_logger().error(f"Joint callback error: {e}")

    def _gripper_cb(self, msg: Float64) -> None:
        self.target_gripper_position = max(0.0, min(1.0, msg.data))
        self.gripper_received = True

    def _tcp_pose_cb(self, msg: Float64MultiArray) -> None:
        """Receive a Cartesian target pose for the follower.

        Accepts ``Float64MultiArray.data`` of length 6 (``[x, y, z, rx, ry,
        rz]``) or 7 (gripper appended; ignored here since the gripper has its own
        topic). When this pose is fresher than ``stale_timeout``, the servo tick
        uses ``servoL`` instead of ``servoJ`` so the UR controller's firmware
        performs the IK. Listens on ``/mirror/tcp_pose_cmd`` — a *command* topic,
        NOT the informational ``/mirror/tcp_pose`` that source_reader uses to
        echo the leader arm's pose during teach-mode mirroring.
        """
        data = list(msg.data)
        if len(data) < 6:
            self.get_logger().warn(
                f"/mirror/tcp_pose data too short: len={len(data)}")
            return
        new_target = data[:6]
        now = time.monotonic()
        # Shift the current target down to `prev` and update the EMA of
        # inter-arrival interval so the servo-loop interpolator knows how fast to
        # walk from prev -> new across the next interval.
        if self.target_tcp_pose is not None and self.last_tcp_msg_time > 0.0:
            dt = max(1e-3, now - self.last_tcp_msg_time)
            # EMA with alpha=0.3 adapts in a few setpoints; clamp to sensible
            # bounds so a single late sample doesn't blow up the interpolation
            # horizon.
            self._tcp_inter_arrival_ema = max(
                0.02, min(2.0,
                          0.7 * self._tcp_inter_arrival_ema + 0.3 * dt))
            self._tcp_prev_target = list(self.target_tcp_pose)
        else:
            # First setpoint after start (or after a stale gap): no smoothing
            # baseline yet — anchor prev to the new target so the first tick uses
            # servoL straight on the new target instead of jumping from a stale
            # prev.
            self._tcp_prev_target = list(new_target)
        self.target_tcp_pose = new_target
        self._tcp_target_arrival = now
        self.last_tcp_msg_time = now
        self.tcp_msgs_received += 1

    def _di0_cb(self, msg: Bool) -> None:
        self.di0_state = msg.data

    def _gui_toggle_cb(self, msg: Bool) -> None:
        """Handle GUI-triggered start/stop (same as DI0 press)."""
        if msg.data:
            now = time.monotonic()
            if (now - self.last_di0_toggle_time) >= self.DEBOUNCE_SECONDS:
                self.last_di0_toggle_time = now
                self._handle_di0_toggle()
                self.get_logger().info("GUI toggle received")

    # ── State machine helpers ────────────────────────────────────

    def _interpolate_toward(self, target: list[float], max_speed: float) -> float:
        """Move servo_positions toward target at max_speed rad/s.

        Returns worst joint error.
        """
        max_step = max_speed * self.servo_time
        worst = 0.0
        for i in range(6):
            err = target[i] - self.servo_positions[i]
            worst = max(worst, abs(err))
            if abs(err) > max_step:
                if abs(err) > self.max_clamp_excess:
                    self.max_clamp_excess = abs(err)
                self.servo_positions[i] += max_step if err > 0 else -max_step
            else:
                self.servo_positions[i] = target[i]
        return worst

    def _at_home(self) -> bool:
        """Check if servo_positions are within threshold of HOME."""
        if self.servo_positions is None:
            return False
        return all(abs(h - s) < self.alignment_threshold
                   for h, s in zip(self.HOME_POSITIONS, self.servo_positions))

    # ── Main servo tick (125 Hz) ─────────────────────────────────

    def _servo_tick(self) -> None:
        # Hard guard: nothing inside the servo loop is allowed to propagate an
        # exception, otherwise a controller-side protective stop (which makes
        # servoJ/servoL throw) would kill the ROS executor and take the whole
        # app down.
        try:
            self._servo_tick_inner()
        except Exception as e:
            self.get_logger().error(f"_servo_tick crashed: {e}")
            # If it looks like a stop, hand off to the recovery driver.
            self._enter_recovering(f"tick exception: {e}")

    def _servo_tick_inner(self) -> None:
        if not self.motion_confirmed:
            return
        if not self.enable_motion:
            return
        if not self.rtde_connected or self.rtde_control is None:
            return

        # Detect a controller-side protective stop (collision, joint limit, force
        # limit, etc.) BEFORE doing anything else. While latched, all motion is
        # suppressed; recovery is handled by _try_recover_from_protective_stop on
        # the 3 s reconnect tick.
        if self._check_protective_stop():
            return
        if self.state == RECOVERING:
            return

        # Push the desired teach-pendant speed slider to the controller whenever
        # the state or GUI scale implies a different cap. This is the hard,
        # firmware-level limit — every UR motion primitive below honours it
        # regardless of the velocity arg we pass.
        self._apply_speed_slider()

        # Initialise servo_positions once from actual robot pose
        if self.servo_positions is None:
            try:
                self.servo_positions = list(self.rtde_receive.getActualQ())
            except Exception:
                return

        # ── DI0 rising-edge detection ──
        now = time.monotonic()
        if self.di0_state and not self.di0_prev:
            dt = now - self.last_di0_toggle_time
            if dt >= self.DEBOUNCE_SECONDS:
                self.last_di0_toggle_time = now
                self.get_logger().info(
                    f"DI0 rising edge accepted (dt={dt:.3f}s, state={self.state})")
                self._handle_di0_toggle()
            else:
                self.get_logger().warn(
                    f"DI0 rising edge SUPPRESSED by debounce (dt={dt:.3f}s)")
        self.di0_prev = self.di0_state

        # ── State machine ──
        # Compensation factor so alignment / return-to-home keep their intended
        # rad/s rate even though the controller's master speed slider is now
        # active in every phase. The controller scales the executed motion by
        # ``speed_scale``; we pre-divide here so the product lands at
        # ``alignment_speed``. Floor at 0.05 so the divisor never blows up.
        slider_comp = 1.0 / max(0.05, self.speed_scale)

        if self.state == ALIGNING:
            # 1.5625× alignment_speed: home returns are conservative
            # repositioning moves (no payload-tracking, no mirror lag), so we can
            # run them faster than the mirroring catch-up.
            worst = self._interpolate_toward(
                self.HOME_POSITIONS,
                self.alignment_speed * 1.5625 * slider_comp)
            if worst < self.alignment_threshold:
                self.state = IDLE
                self.get_logger().info(
                    "✅ ALIGNMENT complete — at home position. "
                    "Press DI0 on source robot to start mirroring.")
                self._publish_recorder_active(False)

        elif self.state == IDLE:
            # Hold at home (or hold current position if no home)
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
                stale = (self.last_msg_time > 0.0 and
                         (now - self.last_msg_time) > self.stale_timeout)
                if stale:
                    self.stale_ticks += 1
                elif not self.mirroring_aligned:
                    # Catch-up phase: slowly interpolate toward source position
                    worst = self._interpolate_toward(
                        self.source_joint_positions, self.alignment_speed)
                    if worst < self.alignment_threshold:
                        self.mirroring_aligned = True
                        self.get_logger().info(
                            "✅ Catch-up complete — full-speed mirroring active")
                else:
                    # Full-speed tracking — but rate-limit per servo tick so that
                    # a missed source-reader tick (which makes the next delta
                    # unusually large) does not produce a velocity-limit lurch on
                    # the robot. With normal sub-tick deltas the interpolator is a
                    # no-op passthrough, so this only smooths actual bursts. No
                    # pre-scale by ``speed_scale`` — the controller's master
                    # slider applies it for us.
                    self._interpolate_toward(
                        self.source_joint_positions,
                        self.max_velocity)

        elif self.state == RETURNING:
            # 1.5625× alignment_speed: see ALIGNING comment above. The follower
            # is unloaded and tracing a known path back to a known pose, so the
            # faster ramp is safe.
            worst = self._interpolate_toward(
                self.HOME_POSITIONS,
                self.alignment_speed * 1.5625 * slider_comp)
            if worst < self.alignment_threshold:
                self.state = IDLE
                self.get_logger().info(
                    "✅ RETURNED to home — ready for next recording cycle. "
                    "Press DI0 to start.")
                self._publish_recorder_active(False)

        # ── Send servo command ──
        # In MIRRORING, if a fresh /mirror/tcp_pose has been received, send it
        # via ``servoL`` so the UR controller's firmware does the IK from
        # Cartesian to joints. The pose path is opt-in: a publisher must actively
        # send TCP samples, otherwise we fall back to servoJ on the joint target.
        # Outside MIRRORING (i.e. ALIGNING / IDLE / RETURNING) we always use
        # joints because the home pose is joint-defined.
        use_tcp = (
            self.state == MIRRORING
            and self.target_tcp_pose is not None
            and self.last_tcp_msg_time > 0.0
            and (time.monotonic() - self.last_tcp_msg_time)
            <= self.stale_timeout
        )
        try:
            eff_vel = self.max_velocity * self.speed_scale
            eff_acc = self.max_acceleration * self.speed_scale
            if use_tcp:
                # Interpolate prev -> target across the measured inter-arrival
                # interval so the servoL stream is continuous instead of stepping
                # between sparse model setpoints. alpha clamps to [0, 1]: if the
                # next setpoint is late we hold at the current target rather than
                # extrapolating into never-trained territory.
                if self._tcp_prev_target is None:
                    interp_pose = list(self.target_tcp_pose)
                elif self.interp_scale <= 0.0:
                    # Interpolation disabled — send the raw latest setpoint every
                    # tick. Useful for observing the native jerk between model
                    # setpoints.
                    interp_pose = list(self.target_tcp_pose)
                else:
                    dt = time.monotonic() - self._tcp_target_arrival
                    horizon = max(1e-3,
                                  self._tcp_inter_arrival_ema * self.interp_scale)
                    alpha = max(0.0, min(1.0, dt / horizon))
                    prev = self._tcp_prev_target
                    tgt = self.target_tcp_pose
                    interp_pose = [
                        prev[i] + alpha * (tgt[i] - prev[i])
                        for i in range(6)
                    ]
                self.rtde_control.servoL(
                    interp_pose,
                    eff_vel,
                    eff_acc,
                    self.servo_time,
                    self.servo_lookahead,
                    self.servo_gain,
                )
                # Keep ``servo_positions`` in sync with reality so that a
                # fall-back to servoJ (when TCP goes stale) starts from the
                # actual joint pose, not the last sent joint target.
                try:
                    self.servo_positions = list(
                        self.rtde_receive.getActualQ())
                except Exception:
                    pass
            else:
                self.rtde_control.servoJ(
                    self.servo_positions,
                    eff_vel,
                    eff_acc,
                    self.servo_time,
                    self.servo_lookahead,
                    self.servo_gain,
                )
            self.commands_sent += 1

            # Publish destination joint states (always from getActualQ if we used
            # servoL; the commanded ``servo_positions`` is only the source of
            # truth in joint mode).
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(self.JOINT_NAMES)
            js.position = list(self.servo_positions)
            self.joint_state_pub.publish(js)
        except Exception as e:
            mode = "servoL" if use_tcp else "servoJ"
            self.get_logger().error(f"{mode} failed: {e}")
            self.rtde_connected = False

    # ── DI0 toggle handler ───────────────────────────────────────

    def _handle_di0_toggle(self) -> None:
        if self.state == IDLE:
            # Start mirroring + recording. We always publish /recorder/active=True
            # here, even when a policy is driving the destination, because (a)
            # suppressing recording silently confuses users who expect START to
            # record and (b) the /policy/active flag was prone to staying sticky
            # after a model error or unclean stop, which would permanently kill
            # recording until the GUI was restarted. If the user does not want to
            # record policy rollouts, they simply do not press START while the
            # policy is running.
            self.mirroring_aligned = False
            self.state = di0_next_state(IDLE, self.use_home)  # MIRRORING
            self._policy_induced_mirror = False
            self._publish_recorder_active(True)
            note = " (policy_active=True)" if self.policy_active else ""
            self.get_logger().info(
                f"▶ START — catching up to source position...{note}")

        elif self.state == MIRRORING:
            self.state = di0_next_state(MIRRORING, self.use_home)  # RETURNING or IDLE
            self._publish_recorder_active(False)
            if self.use_home:
                self.get_logger().info(
                    "⏹ DI0 pressed — MIRRORING stopped, returning to home. "
                    "Recording OFF.")
            else:
                self.get_logger().info(
                    "⏹ DI0 pressed — MIRRORING stopped, holding position. "
                    "Recording OFF. Press DI0 to resume.")

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
        # Diagnostic: log every transition with caller state so we can explain
        # unexpected episode splits.
        caller = traceback.extract_stack()[-2]
        self.get_logger().info(
            f"/recorder/active <- {active}  (state={self.state}  "
            f"caller={caller.name}:{caller.lineno})")

    # ── Gripper ──────────────────────────────────────────────────

    def _gripper_tick(self) -> None:
        if not self.motion_confirmed:
            return
        if not self.enable_motion:
            return
        if not self.gripper_received:
            return

        # ── Hysteresis + dwell ────────────────────────────────────────
        raw = float(self.target_gripper_position)
        now = time.monotonic()
        new_closed, flipped = gripper_hysteresis(
            raw, self._gripper_is_closed, self._gripper_last_flip_t, now,
            self.gripper_close_threshold, self.gripper_open_threshold,
            self.gripper_min_dwell)
        if flipped:
            self._gripper_is_closed = new_closed
            self._gripper_last_flip_t = now
            self.get_logger().info(
                f'Gripper flip -> {"CLOSED" if new_closed else "OPEN"} '
                f"(raw cmd={raw:.2f})")
        effective = 1.0 if self._gripper_is_closed else 0.0

        # Always publish the effective (binary) state for GUI + recorder so
        # policy rollouts and teach mode see the same convention the actuator is
        # actually executing.
        grip_msg = Float64()
        grip_msg.data = effective
        self.dest_gripper_pub.publish(grip_msg)

        if not self.gripper_connected or self.robotiq is None:
            return
        if abs(effective - self.last_sent_gripper) < 0.05:
            return
        self.last_sent_gripper = effective
        grip_int = max(0, min(255, int(effective * 255)))
        if not self.robotiq.move_to(grip_int, self.gripper_speed, self.gripper_force):
            self.gripper_connected = False

    # ── Destination measured-state publisher ────────────────────

    def _dest_state_tick(self) -> None:
        """Publish the destination robot's actually-achieved joint angles, TCP
        pose and gripper closed-flag at 30 Hz.

        These are the canonical ``observation.*`` signals for the LeRobot
        dataset. Keeping them on dedicated ``/destination/*`` topics avoids
        confusion with ``/joint_states``, which carries the *commanded* servo
        target instead of the achieved state.
        """
        if not self.rtde_connected or self.rtde_receive is None:
            return
        try:
            q = list(self.rtde_receive.getActualQ())
            tcp = list(self.rtde_receive.getActualTCPPose())
        except Exception:
            # Don't spam the log; just drop this tick.
            self.rtde_connected = False
            return

        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = list(self.JOINT_NAMES)
        js.position = q
        self.dest_joint_states_pub.publish(js)

        tp = Float64MultiArray()
        tp.data = tcp
        self.dest_tcp_pose_pub.publish(tp)

        # Gripper closed proxy: target >= 0.5. We don't currently read the
        # Robotiq POS register; this proxy is sample-accurate within one gripper
        # tick (~150 ms) which is below the 15 Hz frame period of the recorder.
        b = Bool()
        b.data = bool(self.target_gripper_position >= 0.5)
        self.dest_gripper_closed_pub.publish(b)

    # ── Logging ──────────────────────────────────────────────────

    def _log_tick(self) -> None:
        m = "ON" if self.enable_motion else "OFF"
        r = "OK" if self.rtde_connected else "DOWN"
        g = "OK" if self.gripper_connected else "DOWN"

        if self.servo_positions:
            pos = ", ".join(f"{p:.3f}" for p in self.servo_positions)
        else:
            pos = "N/A"

        extra = ""
        if self.state in (ALIGNING, RETURNING) and self.servo_positions:
            worst = max(abs(h - s) for h, s in
                        zip(self.HOME_POSITIONS, self.servo_positions))
            deg = math.degrees(worst)
            extra = f" | err={worst:.3f}rad ({deg:.1f}°)"

        self.get_logger().debug(
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

        # Publish state for GUI
        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

    # ── Cleanup ──────────────────────────────────────────────────

    def destroy_node(self) -> None:
        # Signal recording stop before shutdown
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
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    if not _RTDE_AVAILABLE:
        _LOGGER.error("ur_rtde not installed!  pip install ur_rtde")
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
