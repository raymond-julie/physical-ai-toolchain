"""Configuration loading for ur_dual_recorder.

The device topology (arms + cameras) is read from the TrainMyBot config file
(default ``/etc/trainmybot/config_v3.yaml``). That file is owned by the wider
TrainMyBot deployment, so this module treats it as **read-only** and never
writes to it.

Settings that are specific to *this* application — analog->gripper scaling,
servo gains, recording parameters — are not present in ``config_v3.yaml``. They
come from an optional overlay file (``config/app.yaml`` next to the package, or
``--app-config <path>``) merged on top of built-in defaults. This keeps the
shared TrainMyBot config untouched while still allowing per-site tuning
(defaults < app.yaml < CLI).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - yaml is a hard dependency
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

DEFAULT_CONFIG_PATH = "/etc/trainmybot/config_v3.yaml"

UR_JOINT_NAMES: list[str] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Built-in application defaults. Overlaid by the optional app-config file and
# then by CLI flags (defaults < app.yaml < CLI).
DEFAULT_APP_SETTINGS: dict[str, Any] = {
    "control": {
        # servoJ loop rate (Hz). config_v3 control_loop runs at 500 Hz; 125 Hz
        # (8 ms) is the safe, well-tested default. Raise to 250/500 once the
        # network round-trip to the controllers has been verified.
        "frequency": 125.0,
        "servo_time": 0.008,
        "servo_lookahead": 0.05,
        "servo_gain": 200,
        "max_velocity": 1.5,
        "max_acceleration": 3.0,
        # Initial slow alignment of each follower to its leader's pose.
        "alignment_speed": 0.25,
        "alignment_threshold": 0.02,
        # Freeze the servo target if no fresh leader sample within this window.
        "stale_timeout": 0.2,
        # Per-joint soft limits (rad). None -> no clamp.
        "joint_limits": None,
    },
    "gripper": {
        # analog->gripper teleop is shelved in recording-only mode; the follower
        # gripper is read, not driven. These keys remain for the (shelved) teleop
        # path and to configure gripper reads.
        # Robotiq raw position (0..255) at/above which the gripper is "closed".
        "closed_threshold": 128,
        # Which leader analog input carries the squeeze sensor (teleop only).
        #   tool0 | tool1 | standard0 | standard1
        # "tool2" in UR teach-pendant numbering == analog_in[2] == tool0.
        "analog_input": "tool0",
        # Raw analog range (volts in 0..10 V mode, amps in 0.004..0.020 A mode).
        "analog_min": 0.0,
        "analog_max": 10.0,
        # Low signal -> open, high signal -> closed (set True to invert).
        "invert": False,
        # Ignore fractional changes below this to avoid socket spam.
        "deadband": 0.01,
        # Snap-to-extreme bands so the gripper fully opens/closes at the ends.
        "open_band": 0.03,
        "close_band": 0.03,
        "speed": 255,
        "force": 80,
        # Robotiq command rate (Hz).
        "command_rate": 10.0,
        "port": 63352,
    },
    "recording": {
        "enabled": True,
        "repo_id": "local/ur_dual",
        "root": "./recordings_lerobot",
        "fps": 30.0,
        "task": "dual_arm_recording",
        "use_videos": True,
        "min_episode_frames": 5,
        "image_height": 480,
        "image_width": 848,
        "max_buffer_gb": 6.0,
        "defer_encoding": False,
        "video_codec": "h264",
        "lerobot_format": "v2.1",
    },
    "web": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
    },
    "camera": {
        # Where camera frames come from:
        #   "orbbec" — open each Orbbec device directly (default).
        #   "stream" — consume MJPEG feeds from a running camera_streamer so the
        #              recorder can share cameras with another process.
        "source": "orbbec",
        # Base URL of the camera_streamer when source == "stream". Each camera is
        # read from "{base}/stream/{serial}".
        "stream_base_url": "http://127.0.0.1:8000",
        # Optional per-camera URL overrides keyed by config_v3 device_id, e.g.
        #   cam_high: "http://10.50.102.2:8000/stream/CV3H4600001E"
        "stream_urls": {},
    },
    "pedal": {
        # USB foot pedal (e.g. iKKEGOL) that toggles recording on each press.
        "enabled": True,
        # Explicit input device, e.g. "/dev/input/event7". Takes priority.
        "device_path": "",
        # Case-insensitive substring matched against device names when
        # device_path is empty. Empty -> auto-detect by name hints
        # (pedal/foot/switch). Identify yours with `--list-pedals`.
        "device_name": "",
        # Specific key the pedal emits, e.g. "KEY_B". Empty -> any key-down.
        # Only used by the single-pedal toggle fallback when `actions` is empty.
        "key": "",
        # Grab the device exclusively so the keystroke doesn't leak to the OS.
        "grab": True,
        # Map each pedal's key to a recording action. The iKKEGOL 3-pedal switch
        # sends KEY_A / KEY_B / KEY_C by default (left->right). Set a value to ""
        # to leave that pedal unbound.
        #   start   -> start a new episode
        #   stop    -> stop and SAVE the current episode
        #   discard -> stop and DELETE the current episode (no save)
        "actions": {
            "start": "KEY_A",
            "stop": "KEY_B",
            "discard": "KEY_C",
        },
    },
}


class DualRecorderConfigError(RuntimeError):
    """Raised when a recorder config file is missing or has no usable pairs."""


@dataclass
class ArmConfig:
    """A single UR arm from the TrainMyBot config."""

    device_id: str
    name: str
    ip: str
    mode: str  # "leader" | "follower"
    model: str = "ur_ps5"
    side: str = ""  # "left" | "right" (inferred from device_id)


@dataclass
class CameraConfig:
    """A single camera device from the TrainMyBot config."""

    device_id: str
    name: str
    model: str
    serial: str
    resolution: str = "848x480"
    fps: int = 30


@dataclass
class LeaderFollowerPair:
    """A leader arm bound to the follower it commands."""

    side: str
    leader: ArmConfig
    follower: ArmConfig


@dataclass
class AppConfig:
    """Fully-resolved configuration consumed by the application."""

    source_path: Path
    arms: dict[str, ArmConfig]
    cameras: dict[str, CameraConfig]
    pairs: list[LeaderFollowerPair]
    settings: dict[str, Any]
    recordings_dir: str = ""
    copy_target_dir: str = ""

    @property
    def control(self) -> dict[str, Any]:
        return self.settings["control"]

    @property
    def gripper(self) -> dict[str, Any]:
        return self.settings["gripper"]

    @property
    def recording(self) -> dict[str, Any]:
        return self.settings["recording"]

    @property
    def web(self) -> dict[str, Any]:
        return self.settings["web"]

    @property
    def camera(self) -> dict[str, Any]:
        return self.settings["camera"]

    @property
    def pedal(self) -> dict[str, Any]:
        return self.settings["pedal"]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _infer_side(device_id: str) -> str:
    lowered = device_id.lower()
    if "left" in lowered:
        return "left"
    if "right" in lowered:
        return "right"
    return ""


def _parse_devices(
    devices: dict[str, Any],
) -> tuple[dict[str, ArmConfig], dict[str, CameraConfig]]:
    arms = {}
    cameras = {}

    for device_id, spec in (devices or {}).items():
        if not isinstance(spec, dict):
            continue
        dtype = spec.get("type")
        if dtype == "arm":
            ip = spec.get("ip")
            if not ip:
                continue
            arms[device_id] = ArmConfig(
                device_id=device_id,
                name=spec.get("name", device_id),
                ip=str(ip),
                mode=str(spec.get("mode", "")).lower(),
                model=str(spec.get("model", "ur_ps5")),
                side=_infer_side(device_id),
            )
        elif dtype == "camera":
            specs = spec.get("specs", {}) or {}
            default_spec = specs.get("default", {}) or {}
            cameras[device_id] = CameraConfig(
                device_id=device_id,
                name=spec.get("name", device_id),
                model=str(spec.get("model", "")),
                serial=str(spec.get("serial", "")),
                resolution=str(default_spec.get("resolution", "848x480")),
                fps=int(default_spec.get("fps", 30)),
            )

    return arms, cameras


def _build_pairs(arms: dict[str, ArmConfig]) -> list[LeaderFollowerPair]:
    """Pair each leader with the follower on the same side.

    Pairing is keyed on the inferred ``side`` ("left"/"right"). Arms with no
    detectable side are matched as a single anonymous pair when exactly one
    leader and one follower remain.
    """
    leaders = [a for a in arms.values() if a.mode == "leader"]
    followers = [a for a in arms.values() if a.mode == "follower"]

    pairs: list[LeaderFollowerPair] = []
    used_followers = set()

    for leader in leaders:
        match = next(
            (
                f
                for f in followers
                if f.side == leader.side and f.device_id not in used_followers
            ),
            None,
        )
        if match is None:
            # Fall back to any unused follower (single-pair / unsided configs).
            match = next(
                (f for f in followers if f.device_id not in used_followers),
                None,
            )
        if match is None:
            raise DualRecorderConfigError(
                f"Leader '{leader.device_id}' has no matching follower arm."
            )
        used_followers.add(match.device_id)
        side = leader.side or match.side or f"pair{len(pairs)}"
        pairs.append(LeaderFollowerPair(side=side, leader=leader, follower=match))

    if not pairs:
        raise DualRecorderConfigError("No leader/follower pairs found in config.")

    pairs.sort(key=lambda p: p.side)
    return pairs


def load_app_settings(app_config_path: str | None) -> dict[str, Any]:
    """Return DEFAULT_APP_SETTINGS merged with an optional overlay file."""
    settings = copy.deepcopy(DEFAULT_APP_SETTINGS)
    if app_config_path:
        path = Path(app_config_path).expanduser()
        if not path.is_file():
            raise DualRecorderConfigError(f"app-config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            overlay = yaml.safe_load(fh) or {}
        settings = _deep_merge(settings, overlay)
    return settings


def load_config(
    config_path: str = DEFAULT_CONFIG_PATH,
    app_config_path: str | None = None,
) -> AppConfig:
    """Load and resolve the full application configuration.

    Args:
        config_path: TrainMyBot ``config_v3.yaml`` path (device topology).
        app_config_path: Optional overlay file for this app's own settings.

    Raises:
        DualRecorderConfigError: if ``config_path`` does not exist or the config
            has no usable leader/follower pairs.
    """
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise DualRecorderConfigError(
            f"TrainMyBot config not found: {path}. "
            f"Pass --config <path> to point at config_v3.yaml."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    arms, cameras = _parse_devices(raw.get("devices", {}))
    pairs = _build_pairs(arms)
    settings = load_app_settings(app_config_path)

    return AppConfig(
        source_path=path,
        arms=arms,
        cameras=cameras,
        pairs=pairs,
        settings=settings,
        recordings_dir=str(raw.get("recordings_dir", "")),
        copy_target_dir=str(raw.get("copy_target_dir", "")),
    )
