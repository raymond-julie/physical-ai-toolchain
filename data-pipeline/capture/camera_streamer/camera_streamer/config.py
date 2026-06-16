"""Camera configuration for the streamer.

Cameras are selected one of two ways:

1. Auto-discovery (default): query the Orbbec SDK for every connected device and
   stream them all. No config file needed.
2. Device config (optional): point ``--config`` at a ``config_v3.yaml``; only
   the ``type: camera`` devices listed there are published, using their friendly
   names, serials, and resolutions.

Service settings (HTTP host/port, JPEG quality, preview frame rate) come from
built-in defaults overlaid by an optional ``--app-config`` YAML file, then CLI
flags (defaults < app.yaml < CLI).
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

DEFAULT_APP_SETTINGS: dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
    },
    "stream": {
        "jpeg_quality": 80,
        "fps": 30,
        "max_width": 1280,
    },
}


class CameraStreamerConfigError(RuntimeError):
    """Raised when a camera_streamer config file is missing or invalid."""


@dataclass
class CameraConfig:
    """A single camera to publish."""

    device_id: str
    name: str
    serial: str = ""
    model: str = ""
    resolution: str = "848x480"
    fps: int = 30


@dataclass
class AppConfig:
    """Fully-resolved configuration consumed by the service."""

    cameras: dict[str, CameraConfig]
    settings: dict[str, Any]
    source_path: Path | None = None

    @property
    def server(self) -> dict[str, Any]:
        return self.settings["server"]

    @property
    def stream(self) -> dict[str, Any]:
        return self.settings["stream"]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_app_settings(app_config_path: str | None) -> dict[str, Any]:
    """Return DEFAULT_APP_SETTINGS merged with an optional overlay file."""
    settings = copy.deepcopy(DEFAULT_APP_SETTINGS)
    if app_config_path:
        path = Path(app_config_path).expanduser()
        if not path.is_file():
            raise CameraStreamerConfigError(f"app-config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle) or {}
        settings = _deep_merge(settings, overlay)
    return settings


def _parse_cameras(devices: dict[str, Any]) -> dict[str, CameraConfig]:
    cameras: dict[str, CameraConfig] = {}
    for device_id, spec in (devices or {}).items():
        if not isinstance(spec, dict) or spec.get("type") != "camera":
            continue
        specs = spec.get("specs", {}) or {}
        default_spec = specs.get("default", {}) or {}
        cameras[device_id] = CameraConfig(
            device_id=device_id,
            name=spec.get("name", device_id),
            serial=str(spec.get("serial", "")),
            model=str(spec.get("model", "")),
            resolution=str(default_spec.get("resolution", "848x480")),
            fps=int(default_spec.get("fps", 30)),
        )
    return cameras


def discover_cameras() -> dict[str, CameraConfig]:
    """Return a CameraConfig for every attached Orbbec device.

    Returns an empty dict when the SDK is unavailable; the caller can fall back
    to a single synthetic camera so the service still starts.
    """
    try:
        from pyorbbecsdk import Context
    except ImportError:
        return {}

    cameras: dict[str, CameraConfig] = {}
    try:
        # Keep the Context alive for the whole enumeration: the device list holds
        # a borrowed pointer to the context's device manager, so collecting a
        # temporary Context() mid-loop crashes with a NULL deviceMgr pointer.
        context = Context()
        device_list = context.query_devices()
        for index in range(device_list.get_count()):
            info = device_list.get_device_by_index(index).get_device_info()
            serial = ""
            model = ""
            try:
                serial = info.get_serial_number() or ""
                model = info.get_name() or ""
            except Exception:
                pass
            device_id = serial or f"camera_{index}"
            name = f"{model} ({serial[-6:]})" if model and serial else (model or f"camera_{index}")
            cameras[device_id] = CameraConfig(
                device_id=device_id,
                name=name,
                serial=serial,
                model=model,
            )
    except Exception:
        return {}
    return cameras


def load_config(
    config_path: str | None = None,
    app_config_path: str | None = None,
) -> AppConfig:
    """Load the service configuration.

    When ``config_path`` is omitted, all connected Orbbec devices are
    auto-discovered. ``app_config_path`` overlays this service's own settings.
    """
    settings = load_app_settings(app_config_path)

    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise CameraStreamerConfigError(
                f"Config not found: {path}. Omit --config to auto-discover cameras."
            )
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        cameras = _parse_cameras(raw.get("devices", {}))
        return AppConfig(cameras=cameras, settings=settings, source_path=path)

    return AppConfig(cameras=discover_cameras(), settings=settings, source_path=None)


def camera_list(config: AppConfig) -> list[CameraConfig]:
    """Return the configured cameras as a list."""
    return list(config.cameras.values())
