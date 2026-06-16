"""CLI entrypoint: open every camera and publish it as an MJPEG link.

    python -m camera_streamer                 # auto-discover, serve on :8000
    python -m camera_streamer --port 9000
    python -m camera_streamer --config /etc/trainmybot/config_v3.yaml
    python -m camera_streamer --list          # list cameras and exit
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import Any

from .cameras import ORBBEC_AVAILABLE, CameraManager
from .config import AppConfig, CameraConfig, load_config
from .server import StreamServer, lan_ip

_LOGGER = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="camera_streamer",
        description="Host connected cameras as shareable MJPEG links on the LAN.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional device config_v3.yaml listing cameras. "
        "Omit to auto-discover all connected Orbbec devices.",
    )
    parser.add_argument(
        "--app-config",
        default=None,
        help="Optional overlay YAML for server/stream settings.",
    )
    parser.add_argument("--host", default=None, help="Bind host (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default 8000).")
    parser.add_argument("--quality", type=int, default=None, help="JPEG quality 1-100.")
    parser.add_argument("--fps", type=float, default=None, help="Max streamed FPS.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Downscale frames wider than this before encoding (0 = off).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the cameras that would be served, then exit.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.host is not None:
        config.server["host"] = args.host
    if args.port is not None:
        config.server["port"] = args.port
    if args.quality is not None:
        config.stream["jpeg_quality"] = args.quality
    if args.fps is not None:
        config.stream["fps"] = args.fps
    if args.max_width is not None:
        config.stream["max_width"] = args.max_width


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, open cameras, and serve the MJPEG streams."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config, args.app_config)
    _apply_overrides(config, args)

    if not config.cameras:
        if not ORBBEC_AVAILABLE:
            _LOGGER.warning(
                "pyorbbecsdk not installed and no --config given; serving one synthetic test camera."
            )
        else:
            _LOGGER.warning("No cameras discovered; serving one synthetic test camera.")
        config.cameras = {
            "synthetic_0": CameraConfig(device_id="synthetic_0", name="Synthetic Camera")
        }

    if args.list:
        print(f"Cameras to serve ({len(config.cameras)}):")
        for cam in config.cameras.values():
            print(f"  - {cam.device_id:24s} {cam.name}  serial={cam.serial or '-'}")
        return 0

    cameras = CameraManager(config.cameras)
    cameras.open_all()
    # Auto-pick-up cameras that appear after startup; only meaningful when
    # auto-discovering devices (no explicit --config).
    if args.config is None:
        cameras.start_supervisor()

    server = StreamServer(config, cameras)
    if not server.start():
        cameras.stop_all()
        return 1

    ip = lan_ip()
    port = int(config.server.get("port", 8000))
    print(f"\nCamera streams are live. Open the dashboard at:\n  http://{ip}:{port}\n")

    stop_event = threading.Event()

    def _handle(_signum: int, _frame: Any) -> None:
        _LOGGER.info("Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    try:
        stop_event.wait()
    finally:
        server.stop()
        cameras.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
