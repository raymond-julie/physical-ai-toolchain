"""Orbbec camera capture for the recorder.

Each configured camera runs a background capture thread that keeps the latest
decoded RGB frame available via :meth:`OrbbecCamera.get_frame`. The recorder and
the web GUI both pull from that single cached frame, so adding consumers never
adds device load.

If the Orbbec SDK (``pyorbbecsdk``) is not importable, or a device fails to open,
the camera degrades to a synthetic test pattern so the rest of the pipeline
(recording schema + preview) can still be exercised without hardware.

The capture path supports the Gemini 305g/335Lg, a GMSL stereo device that
exposes LEFT_COLOR_SENSOR / RIGHT_COLOR_SENSOR rather than a single COLOR_SENSOR.
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

# Color sensor candidates in priority order. The Gemini 305g/335Lg (GMSL stereo)
# exposes LEFT_COLOR_SENSOR / RIGHT_COLOR_SENSOR rather than the single
# COLOR_SENSOR found on standard UVC Orbbec cameras, so we probe in order and use
# the first sensor the device actually provides.
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
            for i in range(sensor_list.get_count()):
                available.add(sensor_list.get_sensor_by_index(i).get_type())
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
        count = device_list.get_count()
        for i in range(count):
            dev = device_list.get_device_by_index(i)
            info = dev.get_device_info()
            if not serial or info.get_serial_number() == serial:
                return dev
        return None

    def _pick_profile(self, profiles: Any) -> Any:
        try:
            return profiles.get_video_stream_profile(
                self.width, self.height, OBFormat.RGB, self.fps
            )
        except Exception:
            return profiles.get_default_video_stream_profile()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.cfg.device_id}", daemon=True
        )
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
        while not self._stop.is_set():
            t0 = time.monotonic()
            frame = self._capture()
            if frame is not None:
                with self._lock:
                    self._frame = frame
            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _capture(self) -> np.ndarray | None:
        if self._synthetic or self._pipeline is None:
            return self._synthetic_frame()
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
        t = time.time()
        h, w = self.height, self.width
        # Diagonal color gradient so the panel is obviously a live test pattern
        # (not a dead/black feed) even without real cameras.
        xs = np.linspace(0, 255, w, dtype=np.uint8)
        ys = np.linspace(0, 255, h, dtype=np.uint8)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = xs[None, :]  # R ramps left->right
        img[..., 1] = ys[:, None]  # G ramps top->bottom
        img[..., 2] = (128 + 127 * np.sin(t)) * np.ones((h, w))  # B pulses
        # Sweeping vertical bar.
        shift = int((t * 120) % w)
        img[:, shift : min(shift + 30, w)] = (255, 255, 255)
        # Label + SYNTHETIC banner so it's clear this is a placeholder feed.
        try:
            import cv2

            cv2.putText(
                img,
                self.cfg.device_id,
                (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "SYNTHETIC (no camera)",
                (12, h - 20),
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


class MjpegCamera:
    """Consumes an MJPEG HTTP stream (e.g. from camera_streamer).

    Lets the recorder share cameras with another process: the streamer owns the
    physical Orbbec device and re-publishes it over HTTP, so the recorder pulls
    frames from the URL instead of opening the device a second time (the same
    camera cannot be opened by two processes at once).

    Exposes the same ``open``/``start``/``stop``/``get_frame`` surface as
    :class:`OrbbecCamera` so :class:`CameraManager` can use either transparently.
    """

    def __init__(self, cfg: CameraConfig, url: str) -> None:
        self.cfg = cfg
        self.url = url
        self.width, self.height = _parse_resolution(cfg.resolution)
        self.fps = cfg.fps
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self.connected = False

    def open(self) -> bool:
        # The stream is opened lazily in the capture thread (with reconnect), so
        # opening here just records intent and validates cv2 availability.
        try:
            import cv2  # noqa: F401
        except ImportError:
            _LOGGER.error(
                "opencv is required to decode MJPEG streams; camera '%s' disabled",
                self.cfg.device_id,
            )
            return False
        _LOGGER.info(
            "Camera '%s' will consume MJPEG stream %s", self.cfg.device_id, self.url
        )
        return True

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"mjpeg-{self.cfg.device_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._consume_stream()
                backoff = 0.5
            except Exception as exc:
                self.connected = False
                if self._stop.is_set():
                    break
                _LOGGER.warning(
                    "Camera '%s' stream error (%s); reconnecting in %.1fs",
                    self.cfg.device_id,
                    exc,
                    backoff,
                )
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)

    def _consume_stream(self) -> None:
        """Read the multipart MJPEG body and decode JPEG frames by SOI/EOI."""
        import urllib.request

        import cv2

        req = urllib.request.Request(self.url, headers={"User-Agent": "ur-recorder"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.connected = True
            buf = bytearray()
            while not self._stop.is_set():
                chunk = resp.read(8192)
                if not chunk:
                    raise ConnectionError("stream ended")
                buf.extend(chunk)
                # Extract the most recent complete JPEG (SOI 0xFFD8 .. EOI 0xFFD9).
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2)
                while start != -1 and end != -1:
                    jpeg = bytes(buf[start : end + 2])
                    del buf[: end + 2]
                    img = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        with self._lock:
                            self._frame = np.ascontiguousarray(rgb, dtype=np.uint8)
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)
                # Guard against unbounded growth if markers never align.
                if len(buf) > 4_000_000:
                    del buf[:-1_000_000]

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()


def _stream_url(base_url: str, cfg: CameraConfig, overrides: dict[str, str]) -> str:
    """Resolve the MJPEG URL for a camera.

    Priority: explicit per-camera override (by device_id) > ``{base}/stream/{serial}``.
    The camera_streamer uses each camera's serial number as its stream id.
    """
    override = overrides.get(cfg.device_id)
    if override:
        return override
    base = base_url.rstrip("/")
    stream_id = cfg.serial or cfg.device_id
    return f"{base}/stream/{stream_id}"


class CameraManager:
    """Opens and runs every configured camera.

    ``camera_settings`` selects the frame source:

    * ``source: "orbbec"`` (default) — open each Orbbec device directly.
    * ``source: "stream"`` — consume MJPEG feeds from a running camera_streamer at
      ``stream_base_url`` (per-camera ``stream_urls`` override the default
      ``{base}/stream/{serial}`` mapping).
    """

    def __init__(
        self,
        cameras: dict[str, CameraConfig],
        camera_settings: dict | None = None,
    ) -> None:
        settings = camera_settings or {}
        source = str(settings.get("source", "orbbec")).lower()
        if source == "stream":
            base_url = str(settings.get("stream_base_url", "http://127.0.0.1:8000"))
            overrides = settings.get("stream_urls", {}) or {}
            self.cameras: dict[str, OrbbecCamera | MjpegCamera] = {
                dev_id: MjpegCamera(cfg, _stream_url(base_url, cfg, overrides))
                for dev_id, cfg in cameras.items()
            }
            _LOGGER.info("Camera source: MJPEG streams from %s", base_url)
        else:
            self.cameras = {
                dev_id: OrbbecCamera(cfg) for dev_id, cfg in cameras.items()
            }

    def open_all(self) -> None:
        for cam in self.cameras.values():
            cam.open()
            cam.start()

    def stop_all(self) -> None:
        for cam in self.cameras.values():
            cam.stop()

    def get_frame(self, device_id: str) -> np.ndarray | None:
        cam = self.cameras.get(device_id)
        return cam.get_frame() if cam else None

    def device_ids(self) -> list[str]:
        return list(self.cameras.keys())
