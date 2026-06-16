"""Per-pair leader->follower teleoperation control.

A :class:`PairController` owns one leader arm, its matching follower arm, and the
follower's Robotiq gripper. It runs a single high-rate control thread that:

1. reads the leader's joint setpoint and streams it to the follower (``servoJ``),
2. reads the leader's tool analog sensor and drives the follower gripper to the
   scaled position.

A slow ``moveJ`` alignment runs once before live mirroring so the follower never
makes a sudden jump toward the leader's pose.

The controller is safe to construct without hardware; ``connect`` simply reports
failure and the loop idles until reconnected.

This module is part of the shelved teleop path and is not imported in
recording-only mode; it remains for when mirroring is re-enabled.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from .analog import AnalogGripperMap
from .config import LeaderFollowerPair
from .robotiq import RobotiqGripper, RobotiqReconnector
from .ur_interface import FollowerInterface, LeaderInterface

_LOGGER = logging.getLogger(__name__)


@dataclass
class PairSnapshot:
    """Latest observation/action data for one pair (consumed by the recorder)."""

    leader_q: list[float] = field(default_factory=lambda: [0.0] * 6)
    follower_q: list[float] = field(default_factory=lambda: [0.0] * 6)
    gripper_cmd: float = 0.0  # normalized 0..1 (closed fraction) — action
    gripper_actual: float = 0.0  # normalized 0..1 from follower — observation
    analog_raw: float = 0.0
    leader_connected: bool = False
    follower_connected: bool = False
    gripper_connected: bool = False


class PairController:
    """Controls one leader->follower pair on its own thread."""

    def __init__(
        self,
        pair: LeaderFollowerPair,
        control_cfg: dict,
        gripper_cfg: dict,
        enable_motion: bool = True,
    ) -> None:
        self.pair = pair
        self.side = pair.side
        self.control_cfg = control_cfg
        self.gripper_cfg = gripper_cfg
        self.enable_motion = enable_motion

        self.leader = LeaderInterface(
            pair.leader.ip, analog_input=gripper_cfg.get("analog_input", "tool0")
        )
        self.follower = FollowerInterface(pair.follower.ip)
        self.gripper = RobotiqGripper(
            pair.follower.ip, port=int(gripper_cfg.get("port", 63352))
        )
        self._gripper_recon = RobotiqReconnector(self.gripper)

        self.analog_map = AnalogGripperMap(
            analog_min=float(gripper_cfg.get("analog_min", 0.0)),
            analog_max=float(gripper_cfg.get("analog_max", 10.0)),
            invert=bool(gripper_cfg.get("invert", False)),
            open_band=float(gripper_cfg.get("open_band", 0.03)),
            close_band=float(gripper_cfg.get("close_band", 0.03)),
        )

        # Control parameters.
        self._freq = float(control_cfg.get("frequency", 125.0))
        self._period = 1.0 / self._freq
        self._servo_time = float(control_cfg.get("servo_time", 0.008))
        self._lookahead = float(control_cfg.get("servo_lookahead", 0.05))
        self._gain = int(control_cfg.get("servo_gain", 200))
        self._max_vel = float(control_cfg.get("max_velocity", 1.5))
        self._max_acc = float(control_cfg.get("max_acceleration", 3.0))
        self._align_speed = float(control_cfg.get("alignment_speed", 0.25))
        self._stale_timeout = float(control_cfg.get("stale_timeout", 0.2))
        self._joint_limits = control_cfg.get("joint_limits")

        self._gripper_period = 1.0 / float(gripper_cfg.get("command_rate", 10.0))
        self._gripper_speed = int(gripper_cfg.get("speed", 255))
        self._gripper_force = int(gripper_cfg.get("force", 80))
        self._gripper_deadband_raw = max(
            1, round(float(gripper_cfg.get("deadband", 0.01)) * 255)
        )

        # Runtime state.
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._mirroring = threading.Event()
        self._aligned = False
        self._lock = threading.Lock()
        self._snapshot = PairSnapshot()
        self._last_servo_q: list[float] | None = None
        self._last_leader_sample_t = 0.0
        self._last_gripper_t = 0.0
        # Tool DI0 edge tracking (start/stop trigger from the leader button).
        self._di0_prev = False
        self.di0_edge = threading.Event()  # set on a rising edge; app clears it

    # ── lifecycle ───────────────────────────────────────────────────────
    def connect(self) -> bool:
        ok_leader = self.leader.connect()
        ok_follower = self.follower.connect() if self.enable_motion else True
        ok_gripper = self.gripper.connect()
        if not ok_gripper:
            _LOGGER.warning("[%s] gripper not connected (will retry)", self.side)
        return ok_leader and ok_follower

    def align(self) -> bool:
        """Slowly move the follower to the leader's current pose."""
        if not self.enable_motion:
            self._aligned = True
            return True
        leader_q = self.leader.read_target_q()
        if leader_q is None:
            _LOGGER.error("[%s] cannot align — no leader pose", self.side)
            return False
        leader_q = self._clamp(leader_q)
        _LOGGER.info("[%s] aligning follower to leader pose (slow moveJ)...", self.side)
        ok = self.follower.move_j(leader_q, speed=self._align_speed)
        if ok:
            self._last_servo_q = list(leader_q)
            self._aligned = True
            _LOGGER.info("[%s] alignment complete", self.side)
        return ok

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"pair-{self.side}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.follower.servo_stop()

    def shutdown(self) -> None:
        self.stop()
        self.follower.disconnect()
        self.leader.disconnect()
        self.gripper.disconnect()

    # ── mirroring gate ──────────────────────────────────────────────────
    def set_mirroring(self, active: bool) -> None:
        if active:
            self._mirroring.set()
        else:
            self._mirroring.clear()

    @property
    def is_mirroring(self) -> bool:
        return self._mirroring.is_set()

    # ── helpers ─────────────────────────────────────────────────────────
    def _clamp(self, q: list[float]) -> list[float]:
        if not self._joint_limits:
            return list(q)
        out = list(q)
        for i, val in enumerate(out):
            if i < len(self._joint_limits) and self._joint_limits[i]:
                lo, hi = self._joint_limits[i]
                out[i] = max(lo, min(hi, val))
        return out

    def snapshot(self) -> PairSnapshot:
        with self._lock:
            return PairSnapshot(
                leader_q=list(self._snapshot.leader_q),
                follower_q=list(self._snapshot.follower_q),
                gripper_cmd=self._snapshot.gripper_cmd,
                gripper_actual=self._snapshot.gripper_actual,
                analog_raw=self._snapshot.analog_raw,
                leader_connected=self.leader.is_connected,
                follower_connected=self.follower.is_connected,
                gripper_connected=self.gripper.is_connected,
            )

    # ── control thread ──────────────────────────────────────────────────
    def _run(self) -> None:
        _LOGGER.info("[%s] control loop started @ %.0f Hz", self.side, self._freq)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._tick(t0)
            except Exception as exc:  # keep the loop alive
                _LOGGER.error("[%s] tick error: %s", self.side, exc)
            elapsed = time.monotonic() - t0
            sleep = self._period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _tick(self, now: float) -> None:
        leader_q = self.leader.read_target_q()
        di0, _di1 = self.leader.read_tool_digital_inputs()
        analog_raw = self.leader.read_analog()

        # Rising-edge detection on the leader tool DI0 (start/stop trigger).
        if di0 and not self._di0_prev:
            self.di0_edge.set()
        self._di0_prev = di0

        # ── follower servo ──
        target_q = self._last_servo_q
        if leader_q is not None:
            leader_q = self._clamp(leader_q)
            self._last_leader_sample_t = now
            if self._mirroring.is_set() and self._aligned:
                target_q = leader_q
        stale = (now - self._last_leader_sample_t) > self._stale_timeout

        if (
            self.enable_motion
            and self._mirroring.is_set()
            and self._aligned
            and target_q is not None
            and not stale
        ):
            self.follower.servo_j(
                target_q,
                self._max_vel,
                self._max_acc,
                self._servo_time,
                self._lookahead,
                self._gain,
            )
            self._last_servo_q = target_q

        # ── follower gripper from leader analog ──
        gripper_cmd_norm = self.analog_map.normalize(analog_raw)
        if (now - self._last_gripper_t) >= self._gripper_period:
            self._last_gripper_t = now
            if self._gripper_recon.ensure() and self._mirroring.is_set():
                pos = round(gripper_cmd_norm * 255.0)
                self.gripper.move_if_changed(
                    pos,
                    min_delta=self._gripper_deadband_raw,
                    speed=self._gripper_speed,
                    force=self._gripper_force,
                )

        # ── publish snapshot ──
        follower_q = self.follower.get_actual_q() if self.enable_motion else None
        gripper_actual = -1
        # Reading the gripper position every tick would saturate the socket; rely
        # on the command as the observed state when not separately polled.
        with self._lock:
            if leader_q is not None:
                self._snapshot.leader_q = leader_q
            if follower_q is not None:
                self._snapshot.follower_q = follower_q
            elif leader_q is not None and not self.enable_motion:
                self._snapshot.follower_q = leader_q
            self._snapshot.gripper_cmd = gripper_cmd_norm
            if gripper_actual >= 0:
                self._snapshot.gripper_actual = gripper_actual / 255.0
            else:
                self._snapshot.gripper_actual = gripper_cmd_norm
            self._snapshot.analog_raw = analog_raw
