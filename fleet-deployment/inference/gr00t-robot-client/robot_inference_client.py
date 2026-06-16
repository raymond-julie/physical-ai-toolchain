#!/usr/bin/env python3
"""Realtime inference client for the dual-arm UR5e GR00T policy.

This is the "nervous system" that closes the loop the Control UI does not: it
captures the four Orbbec camera frames plus both follower arms' joint states,
asks the GR00T policy server (ZMQ) for an action chunk, and streams the
predicted joint targets to the two UR5e arms over RTDE.

Pipeline::

    cameras (ur-camera-streamer :8000) ─┐
                                         ├─► observation ─► GR00T policy :5555
    arms (RTDE getActualQ) ─────────────┘                        │
                                                                 ▼
    UR5e arms ◄──────────── servoJ(joint targets) ◄──── action.robotN_arm (16x6)

Safety first:

* Defaults to **dry-run**: it queries the policy and logs predicted actions but
  does NOT move the robots. Pass ``--execute`` to actually command the arms.
* The first predicted pose is gated against the current measured joints; if the
  policy would jump an arm by more than ``--start-threshold`` rad it refuses to
  start (use ``--allow-jump`` to override deliberately).
* Every commanded step is clamped to at most ``--max-joint-step`` rad away from
  the previous command, capping joint speed so a single bad prediction cannot
  jerk an arm.
* Optional absolute clamp to the per-joint action range from ``metadata.json``.
* ``servoStop`` / ``stopScript`` / ``disconnect`` run on every exit path.

The GR00T ZMQ wire format (msgpack + numpy-as-.npy) is reimplemented here so the
client needs only ``pyzmq``, ``msgpack``, ``numpy``, ``opencv``/``Pillow`` and
``ur_rtde`` -- not the heavy ``isaac-gr00t`` package.

The policy service is a ClusterIP (``gr00t-gr00t-inference:5555``). From a host
outside the cluster, forward it first::

    kubectl port-forward -n default svc/gr00t-gr00t-inference 5555:5555

Example (dry run, the default)::

    python3 robot_inference_client.py \
        --policy-host 127.0.0.1 --policy-port 5555 \
        --camera-url http://192.168.1.10:8000 \
        --task "pick up the red block and place it in the box"

Add ``--execute`` (and confirm the prompt) to drive the real arms.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np
import requests
import zmq

if TYPE_CHECKING:
    from types import FrameType

_LOGGER = logging.getLogger(__name__)

# Defaults that match the recorded dataset / training contract.
# Followers executed at inference. robot1 == left follower, robot2 == right.
# IPs from /etc/trainmybot/config_v3.yaml (arm_left_follower / arm_right_follower).
DEFAULT_ROBOT1_IP = "192.168.1.11"
DEFAULT_ROBOT2_IP = "192.168.1.13"
# color_0..3 -> physical camera SERIAL. The streamer addresses cameras by serial
# (its /api/cameras id == device serial). The serial<->position mapping is
# config-driven on the rig (/etc/trainmybot/config_v3.yaml); these defaults match
# that file's order: color_0=cam_high, color_1=cam_low, color_2=cam_right_wrist,
# color_3=cam_left_wrist. Confirm per setup before executing.
DEFAULT_CAMERA_NAMES = ["CV3H4600001E", "CV3H46000031", "CV34361000HP", "CV34361000J3"]
IMG_HEIGHT = 480
IMG_WIDTH = 848
ARM_DOF = 6
ACTION_HORIZON = 16
# Training cadence. The action horizon (16 steps ~ 1.07 s) implies ~15 fps even
# though some session info.json files report 30; expose it as --control-hz.
DEFAULT_CONTROL_HZ = 15.0
# servoJ tuning (see ur_rtde examples/py/servoj_example.py).
SERVOJ_LOOKAHEAD = 0.1
SERVOJ_GAIN = 300


class PolicyServerError(RuntimeError):
    """Raised when the GR00T policy server is unreachable or returns an error."""


class StartPoseGateError(RuntimeError):
    """Raised when the first predicted pose is too far from the measured joints."""


# ---------------------------------------------------------------------------
# GR00T ZMQ policy client (minimal, no isaac-gr00t dependency)
# ---------------------------------------------------------------------------
def _encode_custom(obj: Any) -> Any:
    """msgpack ``default`` hook matching gr00t.eval.service.MsgSerializer."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    raise TypeError(f"Cannot serialize object of type {type(obj)}")


def _decode_custom(obj: dict) -> Any:
    """msgpack ``object_hook`` matching gr00t.eval.service.MsgSerializer."""
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class PolicyClient:
    """ZMQ REQ client for the GR00T N1.5 inference server."""

    def __init__(self, host: str, port: int, timeout_ms: int = 15000) -> None:
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._connect()

    def _connect(self) -> None:
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def _call(self, endpoint: str, data: dict | None, requires_input: bool) -> dict:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self._socket.send(msgpack.packb(request, default=_encode_custom))
            reply = self._socket.recv()
        except zmq.error.ZMQError as exc:
            # A timed-out REQ socket is stuck; recreate it before re-raising.
            self._socket.close(linger=0)
            self._connect()
            raise PolicyServerError(f"policy server call '{endpoint}' failed: {exc}") from exc
        response = msgpack.unpackb(reply, object_hook=_decode_custom, raw=False, strict_map_key=False)
        if isinstance(response, dict) and "error" in response:
            raise PolicyServerError(f"policy server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self._call("ping", None, requires_input=False)
            return True
        except PolicyServerError:
            return False

    def get_action(self, observation: dict) -> dict:
        return self._call("get_action", observation, requires_input=True)

    def close(self) -> None:
        self._socket.close(linger=0)


# ---------------------------------------------------------------------------
# Camera source (ur-camera-streamer HTTP snapshots)
# ---------------------------------------------------------------------------
def _decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to an HxWx3 uint8 RGB array."""
    try:
        import cv2

        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2 failed to decode JPEG")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image  # Fallback when OpenCV is unavailable.

        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


class CameraSource:
    """Pulls JPEG snapshots for the four policy cameras (color_0..3)."""

    def __init__(self, base_url: str, camera_ids: list[str], timeout: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._camera_ids = camera_ids
        self._timeout = timeout
        self._session = requests.Session()
        # Grab the four cameras concurrently; sequential GETs would inflate the
        # per-cycle inference gap during which the arms receive no new command.
        self._pool = ThreadPoolExecutor(max_workers=len(camera_ids) or 1)

    @staticmethod
    def list_cameras(base_url: str, timeout: float = 5.0) -> list[dict]:
        resp = requests.get(f"{base_url.rstrip('/')}/api/cameras", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("cameras", [])

    def verify(self) -> None:
        """Confirm all configured cameras exist and are connected."""
        available = {c.get("id"): c for c in self.list_cameras(self._base_url, self._timeout)}
        missing = [cid for cid in self._camera_ids if cid not in available]
        if missing:
            raise PolicyServerError(
                f"cameras {missing} not found on streamer {self._base_url}; available ids: {sorted(available)}"
            )
        for cid in self._camera_ids:
            if not available[cid].get("connected", False):
                _LOGGER.warning("camera %s reports not connected", cid)

    def snapshot(self, camera_id: str) -> np.ndarray:
        resp = self._session.get(f"{self._base_url}/snapshot/{camera_id}", timeout=self._timeout)
        resp.raise_for_status()
        img = _decode_jpeg(resp.content)
        if img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            _LOGGER.debug("camera %s returned %s, expected %s", camera_id, img.shape, (IMG_HEIGHT, IMG_WIDTH, 3))
        return img

    def grab_all(self) -> list[np.ndarray]:
        # map preserves the color_0..3 order of self._camera_ids.
        return list(self._pool.map(self.snapshot, self._camera_ids))


# ---------------------------------------------------------------------------
# Dual-arm RTDE interface
# ---------------------------------------------------------------------------
@dataclass
class ArmHandles:
    control: Any
    receive: Any
    ip: str


class DualArm:
    """Wraps two UR5e arms for joint reads and servoJ streaming."""

    def __init__(self, robot1_ip: str, robot2_ip: str, frequency: float, execute: bool) -> None:
        self._execute = execute
        self._frequency = frequency
        self._arm1: ArmHandles | None = None
        self._arm2: ArmHandles | None = None
        # Dry-run still needs joint reads to build observations & gate jumps.
        self._connect(robot1_ip, robot2_ip, control=execute)

    def _connect(self, robot1_ip: str, robot2_ip: str, control: bool = True) -> None:
        from rtde_receive import RTDEReceiveInterface as RTDEReceive

        c1 = c2 = None
        if control:
            from rtde_control import RTDEControlInterface as RTDEControl

            c1 = RTDEControl(robot1_ip, self._frequency)
            c2 = RTDEControl(robot2_ip, self._frequency)
        r1 = RTDEReceive(robot1_ip, self._frequency)
        r2 = RTDEReceive(robot2_ip, self._frequency)
        self._arm1 = ArmHandles(control=c1, receive=r1, ip=robot1_ip)
        self._arm2 = ArmHandles(control=c2, receive=r2, ip=robot2_ip)
        _LOGGER.info("connected to arms: robot1=%s robot2=%s (control=%s)", robot1_ip, robot2_ip, control)

    def read_joints(self) -> tuple[np.ndarray, np.ndarray]:
        q1 = np.asarray(self._arm1.receive.getActualQ()[:ARM_DOF], dtype=np.float32)
        q2 = np.asarray(self._arm2.receive.getActualQ()[:ARM_DOF], dtype=np.float32)
        return q1, q2

    def servo(self, target1: np.ndarray, target2: np.ndarray, dt: float) -> None:
        """Issue one synchronized servoJ command to both arms (no-op in dry-run).

        No internal pacing: the caller spaces commands at the action cadence
        (1/control_hz). servoJ's ``time`` argument equals that cadence so the
        controller interpolates to each target over one action step.
        """
        if not self._execute:
            return
        self._arm1.control.servoJ(list(map(float, target1)), 0.0, 0.0, dt, SERVOJ_LOOKAHEAD, SERVOJ_GAIN)
        self._arm2.control.servoJ(list(map(float, target2)), 0.0, 0.0, dt, SERVOJ_LOOKAHEAD, SERVOJ_GAIN)

    def stop(self) -> None:
        for arm in (self._arm1, self._arm2):
            if arm is None:
                continue
            try:
                if arm.control is not None:
                    arm.control.servoStop()
                    arm.control.stopScript()
                    arm.control.disconnect()
                arm.receive.disconnect()
            except Exception:
                # Best-effort cleanup on shutdown.
                _LOGGER.exception("error while disconnecting arm %s", arm.ip)


# ---------------------------------------------------------------------------
# Action safety helpers
# ---------------------------------------------------------------------------
def load_action_bounds(metadata_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Read per-joint action min/max from a GR00T metadata.json for clamping."""
    try:
        with open(metadata_path) as fh:
            meta = json.load(fh)
    except (OSError, ValueError) as exc:
        _LOGGER.warning("could not read metadata bounds from %s: %s", metadata_path, exc)
        return None
    # Top key is the embodiment tag (e.g. "new_embodiment").
    embodiment = next(iter(meta))
    stats = meta[embodiment]["statistics"]["action"]
    bounds = {}
    for key in ("robot1_arm", "robot2_arm"):
        lo = np.asarray(stats[key]["min"], dtype=np.float32)
        hi = np.asarray(stats[key]["max"], dtype=np.float32)
        bounds[key] = (lo, hi)
    _LOGGER.info("loaded action joint bounds from %s (embodiment '%s')", metadata_path, embodiment)
    return bounds


def clamp_step(target: np.ndarray, reference: np.ndarray, max_step: float) -> np.ndarray:
    """Limit per-joint motion so no single step exceeds ``max_step`` rad."""
    return reference + np.clip(target - reference, -max_step, max_step)


def clamp_absolute(target: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, margin: float) -> np.ndarray:
    """Clamp ``target`` to the trained action range plus ``margin`` rad of slack."""
    if bounds is None:
        return target
    lo, hi = bounds
    return np.clip(target, lo - margin, hi + margin)


def first_pose_jump(
    first_action1: np.ndarray, first_action2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> float:
    """Max per-joint distance (rad) between the first predicted pose and the measured joints."""
    jump1 = float(np.max(np.abs(first_action1 - q1)))
    jump2 = float(np.max(np.abs(first_action2 - q2)))
    return max(jump1, jump2)


# ---------------------------------------------------------------------------
# Observation assembly
# ---------------------------------------------------------------------------
def build_observation(frames: list[np.ndarray], q1: np.ndarray, q2: np.ndarray, task: str) -> dict:
    """Pack one observation in the GR00T modality format (T=1 leading axis)."""
    obs: dict[str, Any] = {}
    for idx, frame in enumerate(frames):
        obs[f"video.color_{idx}"] = frame[np.newaxis, ...].astype(np.uint8)
    obs["state.robot1_arm"] = q1.reshape(1, ARM_DOF).astype(np.float32)
    obs["state.robot2_arm"] = q2.reshape(1, ARM_DOF).astype(np.float32)
    obs["annotation.human.action.task_description"] = [task]
    return obs


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------
class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.running = True
        self.policy = PolicyClient(args.policy_host, args.policy_port, args.timeout_ms)
        self.cameras = CameraSource(args.camera_url, args.camera_ids)
        self.bounds = load_action_bounds(args.metadata) if args.metadata else None
        self.arms: DualArm | None = None  # connected in run() after preflight

    def preflight(self) -> None:
        _LOGGER.info("pinging policy server %s:%s", self.args.policy_host, self.args.policy_port)
        deadline = time.monotonic() + max(0.0, self.args.wait_for_server)
        attempt = 0
        while True:
            if self.policy.ping():
                break
            attempt += 1
            if time.monotonic() >= deadline:
                raise PolicyServerError(
                    "policy server did not respond to ping; is it running and reachable? "
                    "(the inference Deployment may be scaled to 0 -- press Start in the "
                    "control UI, or raise --wait-for-server)"
                )
            _LOGGER.info(
                "policy server not ready (attempt %d), retrying in 5s (waiting up to %.0fs)",
                attempt,
                self.args.wait_for_server,
            )
            time.sleep(5.0)
        _LOGGER.info("policy server is alive")
        _LOGGER.info("verifying cameras on %s", self.args.camera_url)
        self.cameras.verify()
        _LOGGER.info("cameras OK: color_0..3 -> %s", self.args.camera_ids)

    def _execute_chunk(
        self, a1: np.ndarray, a2: np.ndarray, last1: np.ndarray, last2: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        steps = min(self.args.exec_steps, a1.shape[0])
        for k in range(steps):
            if not self.running:
                break
            t0 = time.monotonic()
            tgt1 = clamp_absolute(a1[k], self.bounds["robot1_arm"] if self.bounds else None, self.args.bounds_margin)
            tgt2 = clamp_absolute(a2[k], self.bounds["robot2_arm"] if self.bounds else None, self.args.bounds_margin)
            tgt1 = clamp_step(tgt1, last1, self.args.max_joint_step)
            tgt2 = clamp_step(tgt2, last2, self.args.max_joint_step)
            self.arms.servo(tgt1, tgt2, dt)  # no-op in dry run
            last1, last2 = tgt1, tgt2
            # Pace at the trained action cadence (control_hz). Without this the
            # RTDE initPeriod/waitPeriod ran the whole chunk in ~16ms, so the
            # arm got a burst of targets then froze until the next inference --
            # the "twitch back and forth" symptom. Sleeping the remainder of dt
            # replays the 16-step chunk at the rate it was recorded.
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
        return last1, last2

    def run(self) -> int:
        self.preflight()
        self.arms = DualArm(self.args.robot1_ip, self.args.robot2_ip, self.args.frequency, self.args.execute)
        dt = 1.0 / self.args.control_hz

        # Gate the first predicted pose against the measured joints.
        q1, q2 = self.arms.read_joints()
        frames = self.cameras.grab_all()
        action = self.policy.get_action(build_observation(frames, q1, q2, self.args.task))
        a1 = np.asarray(action["action.robot1_arm"], dtype=np.float32)
        a2 = np.asarray(action["action.robot2_arm"], dtype=np.float32)
        _LOGGER.info("first action shapes: robot1=%s robot2=%s", a1.shape, a2.shape)
        jump = first_pose_jump(a1[0], a2[0], q1, q2)
        _LOGGER.info("first-step jump from current joints: %.3f rad", jump)
        if jump > self.args.start_threshold and not self.args.allow_jump:
            raise StartPoseGateError(
                f"first predicted pose is {jump:.3f} rad from the current joints "
                f"(> --start-threshold {self.args.start_threshold}). Refusing to start. "
                "Move the arms near a demonstrated start pose, or pass --allow-jump to override."
            )

        if self.args.once:
            _LOGGER.info("--once set: single query complete, not entering control loop")
            return 0

        if self.args.execute:
            _LOGGER.warning("EXECUTE MODE: arms WILL move. Streaming at %.1f Hz", self.args.control_hz)
        else:
            _LOGGER.info("DRY RUN: logging predicted actions only, arms will NOT move")

        last1, last2 = q1, q2
        loop = 0
        while self.running:
            t0 = time.monotonic()
            last1, last2 = self._execute_chunk(a1, a2, last1, last2, dt)
            if not self.running:
                break
            q1, q2 = self.arms.read_joints()
            frames = self.cameras.grab_all()
            action = self.policy.get_action(build_observation(frames, q1, q2, self.args.task))
            a1 = np.asarray(action["action.robot1_arm"], dtype=np.float32)
            a2 = np.asarray(action["action.robot2_arm"], dtype=np.float32)
            loop += 1
            if loop % 10 == 0:
                _LOGGER.info(
                    "loop %d: %.1f ms/chunk, robot1[0]=%s",
                    loop,
                    (time.monotonic() - t0) * 1e3,
                    np.round(a1[0], 3).tolist(),
                )
        return 0

    def shutdown(self) -> None:
        self.running = False
        if self.arms is not None:
            self.arms.stop()
        self.policy.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime dual-arm UR5e inference client for the GR00T policy server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Policy server
    parser.add_argument(
        "--policy-host", default="127.0.0.1", help="GR00T ZMQ policy server host (port-forward target)."
    )
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=15000, help="Per-request ZMQ timeout.")
    # Cameras
    parser.add_argument(
        "--camera-url", default="http://127.0.0.1:8000", help="Base URL of the ur-camera-streamer service."
    )
    parser.add_argument(
        "--camera-ids",
        nargs=4,
        metavar=("COLOR0", "COLOR1", "COLOR2", "COLOR3"),
        default=None,
        help="The four streamer camera ids mapped to color_0..color_3 (must match training order). "
        f"Defaults to logical names {DEFAULT_CAMERA_NAMES}; override with the real serials.",
    )
    # Robots
    parser.add_argument("--robot1-ip", default=DEFAULT_ROBOT1_IP, help="Left follower IP.")
    parser.add_argument("--robot2-ip", default=DEFAULT_ROBOT2_IP, help="Right follower IP.")
    parser.add_argument(
        "--frequency", type=float, default=500.0, help="RTDE control frequency (Hz). 500 for e-Series UR5e."
    )
    # Task / policy IO
    parser.add_argument(
        "--task",
        default="",
        help="Language task description fed to the policy (must match the annotation used in training).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to GR00T metadata.json; enables absolute joint clamping to the trained action range.",
    )
    # Control / timing
    parser.add_argument(
        "--control-hz", type=float, default=DEFAULT_CONTROL_HZ, help="Action playback rate (match training fps)."
    )
    parser.add_argument(
        "--exec-steps",
        type=int,
        default=8,
        help=f"Steps of each {ACTION_HORIZON}-step chunk to execute before re-querying (receding horizon).",
    )
    # Safety
    parser.add_argument(
        "--execute", action="store_true", help="Actually command the arms. Without this it is a dry run."
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.03,
        help="Max per-joint change per control step (rad), caps speed.",
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=0.30,
        help="Refuse to start if the first predicted pose is farther than this (rad) from the measured joints.",
    )
    parser.add_argument(
        "--allow-jump", action="store_true", help="Override the first-pose start gate (use with caution)."
    )
    parser.add_argument(
        "--bounds-margin",
        type=float,
        default=0.10,
        help="Slack (rad) added to metadata action bounds when clamping.",
    )
    parser.add_argument(
        "--wait-for-server",
        type=float,
        default=0.0,
        help="Seconds to keep retrying the policy-server ping before giving up. Useful in-cluster when the "
        "server may be scaled to 0; 0 means fail fast.",
    )
    parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Skip the interactive MOVE confirmation for --execute. Required for unattended/in-cluster execute "
        "(no TTY). Use only when the workspace is known clear.",
    )
    parser.add_argument(
        "--once", action="store_true", help="Do a single ping+query, print shapes, then exit (no loop)."
    )
    parser.add_argument("--ping", action="store_true", help="Only ping the policy server and exit.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.camera_ids is None:
        args.camera_ids = list(DEFAULT_CAMERA_NAMES)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.task:
        _LOGGER.warning(
            "no --task given; the policy was trained with a task description "
            "and may behave poorly with an empty prompt"
        )

    if args.ping:
        client = PolicyClient(args.policy_host, args.policy_port, args.timeout_ms)
        ok = client.ping()
        client.close()
        _LOGGER.info("policy server ping: %s", "OK" if ok else "FAILED")
        return 0 if ok else 1

    if args.execute:
        _LOGGER.warning("=== EXECUTE MODE REQUESTED: the robots WILL move ===")
        if args.assume_yes:
            _LOGGER.warning("--assume-yes set: skipping interactive confirmation")
        else:
            try:
                reply = input("Type 'MOVE' to confirm you have a clear workspace and e-stop ready: ")
            except EOFError:
                reply = ""
            if reply.strip() != "MOVE":
                _LOGGER.error("confirmation not given; aborting")
                return 1

    controller = Controller(args)

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        _LOGGER.info("signal %s received, stopping", signum)
        controller.running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        return controller.run()
    except Exception:
        # Log then clean up the robots on any failure path.
        _LOGGER.exception("controller failed")
        return 1
    finally:
        controller.shutdown()


if __name__ == "__main__":
    sys.exit(main())
