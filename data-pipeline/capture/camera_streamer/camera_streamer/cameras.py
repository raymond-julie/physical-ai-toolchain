"""Orbbec camera capture for the streamer.

Each camera runs a background capture thread that keeps the latest decoded RGB
frame available via :meth:`OrbbecCamera.get_frame`. Every HTTP stream consumer
pulls from that single cached frame, so adding viewers never adds device load.

If the Orbbec SDK (``pyorbbecsdk``) is unavailable, or a device fails to open,
the camera degrades to a synthetic test pattern so the service still starts and
the wiring (discovery -> stream -> browser) can be verified without hardware.

The capture path supports the Gemini 305g, a stereo device that exposes
LEFT_COLOR_SENSOR / RIGHT_COLOR_SENSOR rather than a single COLOR_SENSOR.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any

import numpy as np

from .config import CameraConfig

_LOGGER = logging.getLogger(__name__)

try:
    from pyorbbecsdk import (
        Config,
        Context,
        OBFormat,
        OBFrameType,
        OBSensorType,
        Pipeline,
    )

    ORBBEC_AVAILABLE = True
except ImportError:  # pragma: no cover - hardware dependency
    ORBBEC_AVAILABLE = False

if ORBBEC_AVAILABLE:
    _COLOR_SENSORS = (
        (OBSensorType.COLOR_SENSOR, OBFrameType.COLOR_FRAME),
        (OBSensorType.LEFT_COLOR_SENSOR, OBFrameType.LEFT_COLOR_FRAME),
        (OBSensorType.RIGHT_COLOR_SENSOR, OBFrameType.RIGHT_COLOR_FRAME),
    )
else:  # pragma: no cover - hardware dependency
    _COLOR_SENSORS = ()


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        width, height = resolution.lower().split("x")[:2]
        return int(width), int(height)
    except (ValueError, AttributeError):
        return 848, 480


def _decode_color_frame(frame: Any) -> np.ndarray | None:
    """Best-effort decode of an Orbbec color frame to an HxWx3 RGB array."""
    try:
        import cv2
    except ImportError:
        cv2 = None

    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB:
        return data.reshape((height, width, 3))
    if cv2 is None:
        return None
    if fmt == OBFormat.BGR:
        return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_BGR2RGB)
    if fmt == OBFormat.MJPG:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    if fmt == OBFormat.YUYV:
        yuv = data.reshape((height, width, 2))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_YUYV)
    return None


class OrbbecCamera:
    """Single camera capture thread with a cached latest frame."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.width, self.height = _parse_resolution(cfg.resolution)
        self.fps = cfg.fps
        self._pipeline: Any = None
        self._color_frame_type: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._synthetic = False
        self.connected = False

    def open(self) -> bool:
        if not ORBBEC_AVAILABLE:
            _LOGGER.warning(
                "pyorbbecsdk not installed; camera '%s' uses synthetic frames",
                self.cfg.device_id,
            )
            self._synthetic = True
            return False
        try:
            device = self._find_device(self.cfg.serial)
            if device is None:
                _LOGGER.error(
                    "Orbbec device serial '%s' (%s) not found; using synthetic",
                    self.cfg.serial,
                    self.cfg.device_id,
                )
                self._synthetic = True
                return False
            pipeline = Pipeline(device)
            config = Config()
            sensor_type, frame_type = self._pick_color_sensor(device)
            if sensor_type is None:
                _LOGGER.error(
                    "Camera '%s' exposes no color sensor; using synthetic",
                    self.cfg.device_id,
                )
                self._synthetic = True
                return False
            profiles = pipeline.get_stream_profile_list(sensor_type)
            profile = self._pick_profile(profiles)
            config.enable_stream(profile)
            pipeline.start(config)
            self._pipeline = pipeline
            self._color_frame_type = frame_type
            self.connected = True
            _LOGGER.info(
                "Camera '%s' opened (%dx%d@%d, serial %s)",
                self.cfg.device_id,
                self.width,
                self.height,
                self.fps,
                self.cfg.serial,
            )
            return True
        except Exception as exc:
            _LOGGER.error("Camera '%s' open failed: %s", self.cfg.device_id, exc)
            self._synthetic = True
            return False

    @staticmethod
    def _pick_color_sensor(device: Any) -> tuple[Any, Any]:
        """Return the (sensor_type, frame_type) for the device's color stream."""
        available = set()
        try:
            sensor_list = device.get_sensor_list()
            for index in range(sensor_list.get_count()):
                available.add(sensor_list.get_sensor_by_index(index).get_type())
        except Exception:
            available = set()
        for sensor_type, frame_type in _COLOR_SENSORS:
            if not available or sensor_type in available:
                return sensor_type, frame_type
        return None, None

    @staticmethod
    def _find_device(serial: str) -> Any:
        context = Context()
        device_list = context.query_devices()
        for index in range(device_list.get_count()):
            dev = device_list.get_device_by_index(index)
            info = dev.get_device_info()
            if not serial or info.get_serial_number() == serial:
                return dev
        return None

    def _pick_profile(self, profiles: Any) -> Any:
        try:
            return profiles.get_video_stream_profile(self.width, self.height, OBFormat.RGB, self.fps)
        except Exception:
            return profiles.get_default_video_stream_profile()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.cfg.device_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.stop()
            self._pipeline = None

    def _run(self) -> None:
        period = 1.0 / max(1, self.fps)
        # If a live device stops delivering frames (e.g. a GMSL "VIDIOC_DQBUF
        # failed" the SDK cannot auto-recover from), reopen the pipeline rather
        # than serve a frozen last frame forever.
        stale_after = 3.0
        last_ok = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic()
            frame = self._capture()
            if frame is not None:
                with self._lock:
                    self._frame = frame
                last_ok = t0
            elif not self._synthetic and self._pipeline is not None and (t0 - last_ok) > stale_after:
                _LOGGER.warning(
                    "Camera '%s' stalled %.1fs; reopening stream",
                    self.cfg.device_id,
                    t0 - last_ok,
                )
                if self._reopen():
                    last_ok = time.monotonic()
                else:
                    self._stop.wait(2.0)
                    last_ok = time.monotonic()
            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _reopen(self) -> bool:
        """Tear down and reopen the device pipeline after a stream stall."""
        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.stop()
            self._pipeline = None
        self.connected = False
        # open() sets _synthetic on failure; clear it so a transient stall does
        # not permanently degrade a real device to synthetic frames.
        self._synthetic = False
        ok = self.open()
        if not ok:
            self._synthetic = False
        return ok

    def _capture(self) -> np.ndarray | None:
        if self._synthetic:
            return self._synthetic_frame()
        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(100)
            if frames is None:
                return None
            color = self._extract_color_frame(frames)
            if color is None:
                return None
            img = _decode_color_frame(color)
            if img is None:
                return None
            if img.shape[0] != self.height or img.shape[1] != self.width:
                img = self._resize(img)
            return np.ascontiguousarray(img, dtype=np.uint8)
        except Exception as exc:
            _LOGGER.debug("Camera '%s' capture error: %s", self.cfg.device_id, exc)
            return None

    def _extract_color_frame(self, frames: Any) -> Any:
        """Pull the color video frame from a frameset across camera variants."""
        if self._color_frame_type is not None:
            frame = frames.get_frame(self._color_frame_type)
            if frame is not None:
                try:
                    return frame.as_video_frame()
                except Exception:
                    return frame
        return frames.get_color_frame()

    def _resize(self, img: np.ndarray) -> np.ndarray:
        try:
            import cv2

            return cv2.resize(img, (self.width, self.height))
        except Exception:
            return img

    def _synthetic_frame(self) -> np.ndarray:
        now = time.time()
        height, width = self.height, self.width
        xs = np.linspace(0, 255, width, dtype=np.uint8)
        ys = np.linspace(0, 255, height, dtype=np.uint8)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[..., 0] = xs[None, :]
        img[..., 1] = ys[:, None]
        img[..., 2] = (128 + 127 * np.sin(now)) * np.ones((height, width))
        shift = int((now * 120) % width)
        img[:, shift : min(shift + 30, width)] = (255, 255, 255)
        try:
            import cv2

            cv2.putText(
                img, self.cfg.name, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                img,
                "SYNTHETIC (no camera)",
                (12, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass
        return img

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class CameraManager:
    """Opens and runs every configured camera.

    Optionally supervises the device set: a background thread periodically
    re-discovers connected Orbbec cameras and brings up any that appear after
    startup (e.g. a camera that dropped off the GMSL bus and was reset).
    """

    def __init__(self, cameras: dict[str, CameraConfig]) -> None:
        self._lock = threading.Lock()
        self.cameras: dict[str, OrbbecCamera] = {
            dev_id: OrbbecCamera(cfg) for dev_id, cfg in cameras.items()
        }
        self._supervisor: threading.Thread | None = None
        self._supervisor_stop = threading.Event()

    def open_all(self) -> None:
        for cam in list(self.cameras.values()):
            cam.open()
            cam.start()

    def start_supervisor(self, interval: float = 15.0) -> None:
        """Periodically re-discover cameras and add any that newly appear."""
        if self._supervisor is not None:
            return
        self._supervisor_stop.clear()
        self._supervisor = threading.Thread(
            target=self._supervise, args=(interval,), name="cam-supervisor", daemon=True
        )
        self._supervisor.start()

    def _supervise(self, interval: float) -> None:
        from .config import discover_cameras

        while not self._supervisor_stop.wait(interval):
            try:
                discovered = discover_cameras()
            except Exception as exc:
                _LOGGER.debug("Camera rediscovery failed: %s", exc)
                continue
            for dev_id, cfg in discovered.items():
                with self._lock:
                    if dev_id in self.cameras:
                        continue
                cam = OrbbecCamera(cfg)
                cam.open()
                cam.start()
                with self._lock:
                    self.cameras[dev_id] = cam
                _LOGGER.info("Camera '%s' appeared; now streaming", dev_id)

    def stop_all(self) -> None:
        self._supervisor_stop.set()
        if self._supervisor is not None:
            self._supervisor.join(timeout=2.0)
            self._supervisor = None
        for cam in list(self.cameras.values()):
            cam.stop()

    def get_frame(self, device_id: str) -> np.ndarray | None:
        with self._lock:
            cam = self.cameras.get(device_id)
        return cam.get_frame() if cam else None

    def device_ids(self) -> list[str]:
        with self._lock:
            return list(self.cameras.keys())

    def info(self, device_id: str) -> CameraConfig | None:
        with self._lock:
            cam = self.cameras.get(device_id)
        return cam.cfg if cam else None

    def is_connected(self, device_id: str) -> bool:
        with self._lock:
            cam = self.cameras.get(device_id)
        return bool(cam and cam.connected)
