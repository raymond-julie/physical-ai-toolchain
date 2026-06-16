"""Command-line entrypoint: ``python -m ur_dual_recorder``.

Recording-only pipeline. Loads the configuration, opens the Orbbec cameras and
follower-arm state readers, brings up the web GUI, then records episodes until
Ctrl+C. No robot is commanded.

Recording is toggled by the web Record button or a tool DI0 button on the first
follower arm (disable with ``--no-di0-trigger``).
"""

from __future__ import annotations

import argparse
import logging
import signal

from .app import DualRecorderApp
from .config import DEFAULT_CONFIG_PATH, DualRecorderConfigError, load_config
from .pedal import FootPedalListener, list_input_devices
from .web.server import WebServer

_LOGGER = logging.getLogger("ur_dual_recorder")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ur_dual_recorder",
        description="Dual follower-arm + Orbbec camera recorder (recording-only).",
    )
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"TrainMyBot config_v3.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument(
        "--app-config",
        default=None,
        help="Optional overlay YAML for app settings (recording, gripper).",
    )
    p.add_argument(
        "--no-record", action="store_true", help="Disable episode recording."
    )
    p.add_argument(
        "--no-web", action="store_true", help="Disable the web dashboard."
    )
    p.add_argument(
        "--no-di0-trigger",
        action="store_true",
        help="Do not use the follower tool DI0 as a record toggle.",
    )
    p.add_argument(
        "--no-pedal",
        action="store_true",
        help="Disable the USB foot-pedal record toggle.",
    )
    p.add_argument(
        "--pedal-device",
        default=None,
        help="Foot pedal input device path (e.g. /dev/input/event7) "
        "or a name substring; overrides app.yaml.",
    )
    p.add_argument(
        "--list-pedals",
        action="store_true",
        help="List input devices (to identify the foot pedal) and exit.",
    )
    p.add_argument(
        "--web-port", type=int, default=None, help="Override web GUI port."
    )
    p.add_argument(
        "--camera-source",
        choices=["orbbec", "stream"],
        default=None,
        help="Frame source: open Orbbec devices directly ('orbbec') "
        "or consume MJPEG feeds from a camera_streamer ('stream').",
    )
    p.add_argument(
        "--stream-url",
        default=None,
        help="Base URL of the camera_streamer when --camera-source=stream "
        "(e.g. http://127.0.0.1:8000).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def _configure_multiprocessing() -> None:
    """Spawn video-encoder workers from a small, clean 'forkserver'.

    Forking the multi-GB (torch/CUDA-laden) main process for every episode
    encode adds a multi-second UI stall; a forkserver avoids it.
    """
    try:
        import multiprocessing as mp

        mp.set_start_method("forkserver", force=True)
        try:
            mp.set_forkserver_preload(["lerobot.datasets.lerobot_dataset"])
        except Exception as exc:  # preload is best-effort only
            _LOGGER.debug("forkserver preload skipped: %s", exc)
    except (RuntimeError, ValueError) as exc:
        _LOGGER.debug("Could not set forkserver start method: %s", exc)


def _print_pedals() -> None:
    devices = list_input_devices()
    if not devices:
        print("No input devices found (is python-evdev installed?).")
        return
    print("Input devices (look for the one that appears with the pedal):")
    for d in devices:
        keys = "keys" if d["has_keys"] else "    "
        print(f"  [{keys}] {d['path']:<20} {d['name']}")


def _build_pedal(
    app: DualRecorderApp, pedal_cfg: dict, device_arg: str | None
) -> FootPedalListener:
    device_path = pedal_cfg.get("device_path") or None
    device_name = pedal_cfg.get("device_name") or None
    if device_arg:
        # A path-like argument is treated as an explicit device path.
        if device_arg.startswith("/"):
            device_path, device_name = device_arg, None
        else:
            device_path, device_name = None, device_arg

    # Map each pedal key to its recording action (start / stop / discard).
    action_cfg = pedal_cfg.get("actions") or {}
    action_callbacks = {
        "start": app.start_recording,
        "stop": app.stop_recording,
        "discard": app.discard_recording,
    }
    action_labels = {
        "start": "start recording",
        "stop": "stop + save",
        "discard": "stop + delete",
    }
    key_actions = {}
    key_labels = {}
    for action_name, callback in action_callbacks.items():
        key_name = str(action_cfg.get(action_name, "") or "").strip().upper()
        if key_name:
            key_actions[key_name] = callback
            key_labels[key_name] = action_labels[action_name]

    return FootPedalListener(
        app.toggle_recording if not key_actions else None,
        key_actions=key_actions or None,
        key_labels=key_labels or None,
        device_path=device_path,
        device_name=device_name,
        key=pedal_cfg.get("key") or None,
        grab=bool(pedal_cfg.get("grab", True)),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _configure_multiprocessing()

    if args.list_pedals:
        _print_pedals()
        return 0

    try:
        config = load_config(args.config, app_config_path=args.app_config)
    except DualRecorderConfigError as exc:
        _LOGGER.error("Config error: %s", exc)
        return 2

    if args.no_record:
        config.settings["recording"]["enabled"] = False
    if args.web_port is not None:
        config.settings["web"]["port"] = args.web_port
    if args.camera_source is not None:
        config.settings["camera"]["source"] = args.camera_source
    if args.stream_url is not None:
        config.settings["camera"]["source"] = "stream"
        config.settings["camera"]["stream_base_url"] = args.stream_url

    _LOGGER.info("Loaded config: %s", config.source_path)
    followers = sorted(
        (a for a in config.arms.values() if a.mode == "follower"),
        key=lambda a: (a.side, a.device_id),
    )
    if not followers:
        followers = [p.follower for p in config.pairs]
    for a in followers:
        _LOGGER.info("Recording follower arm %s (%s) [%s]", a.device_id, a.ip, a.side)
    _LOGGER.info("Cameras: %s", ", ".join(config.cameras) or "none")

    app = DualRecorderApp(config, enable_di0_trigger=not args.no_di0_trigger)
    app.start()

    if not args.no_web and config.web.get("enabled", True):
        WebServer(app, host=config.web["host"], port=config.web["port"]).start()

    pedal = None
    pedal_cfg = config.pedal
    if not args.no_pedal and pedal_cfg.get("enabled", True):
        pedal = _build_pedal(app, pedal_cfg, args.pedal_device)
        pedal.start()
        app.pedal = pedal

    def _handle_sig(signum: int, _frame: object) -> None:
        _LOGGER.info("Signal %s — stopping.", signum)
        app.request_stop()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    _LOGGER.info(
        "Ready. Toggle recording with the web Record button%s%s.",
        " or follower tool DI0" if not args.no_di0_trigger else "",
        " or the USB foot pedal" if pedal is not None and pedal.connected else "",
    )

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        if pedal is not None:
            pedal.shutdown()
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
