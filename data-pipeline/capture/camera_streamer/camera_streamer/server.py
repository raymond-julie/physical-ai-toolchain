"""HTTP server that publishes each camera as a shareable MJPEG link.

Endpoints:

* ``GET /``                  — dashboard listing every camera + its links.
* ``GET /api/cameras``       — JSON catalog (id, name, stream/snapshot URLs).
* ``GET /stream/<cam_id>``   — live MJPEG (``multipart/x-mixed-replace``).
* ``GET /snapshot/<cam_id>`` — single current JPEG frame.
* ``GET /healthz``           — liveness probe.

One capture thread per camera feeds an unlimited number of HTTP subscribers, so
many viewers on the LAN add no extra device load. The accept loop self-heals.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from .cameras import CameraManager
from .config import AppConfig

if TYPE_CHECKING:
    from flask import Flask, Response

_LOGGER = logging.getLogger(__name__)

try:
    from flask import Flask, Response, abort, jsonify, render_template, request

    FLASK_AVAILABLE = True
except ImportError:  # pragma: no cover
    FLASK_AVAILABLE = False


def lan_ip() -> str:
    """Best-effort primary LAN IPv4 of this host (for printable share links)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are sent; this only selects the outbound interface.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def _encode_jpeg(frame: np.ndarray, quality: int, max_width: int) -> bytes | None:
    try:
        import cv2

        height, width = frame.shape[:2]
        if max_width and width > max_width:
            scale = max_width / float(width)
            frame = cv2.resize(
                frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA
            )
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        return buf.tobytes() if ok else None
    except Exception:
        return None


class StreamServer:
    """Background Flask server publishing every camera as an MJPEG link."""

    def __init__(self, config: AppConfig, cameras: CameraManager) -> None:
        self.config = config
        self.cameras = cameras
        self.host = str(config.server.get("host", "0.0.0.0"))
        self.port = int(config.server.get("port", 8000))
        self.quality = int(config.stream.get("jpeg_quality", 80))
        self.fps = float(config.stream.get("fps", 15))
        self.max_width = int(config.stream.get("max_width", 1280))
        self._thread: threading.Thread | None = None
        self._flask: Flask | None = None
        self._server: Any = None
        self._stop = False

    def _catalog(self, base_url: str) -> list[dict[str, Any]]:
        cams: list[dict[str, Any]] = []
        for cam_id in self.cameras.device_ids():
            cfg = self.cameras.info(cam_id)
            cams.append(
                {
                    "id": cam_id,
                    "name": cfg.name if cfg else cam_id,
                    "serial": cfg.serial if cfg else "",
                    "model": cfg.model if cfg else "",
                    "resolution": cfg.resolution if cfg else "",
                    "connected": self.cameras.is_connected(cam_id),
                    "stream_url": f"{base_url}/stream/{cam_id}",
                    "snapshot_url": f"{base_url}/snapshot/{cam_id}",
                }
            )
        return cams

    def _base_url(self) -> str:
        host = request.host if request else f"{lan_ip()}:{self.port}"
        scheme = request.scheme if request else "http"
        return f"{scheme}://{host}"

    def _build(self) -> Flask:
        flask_app = Flask(__name__, template_folder="templates", static_folder="static")
        server = self

        @flask_app.route("/")
        def index() -> str:
            return render_template("index.html", cameras=server._catalog(server._base_url()))

        @flask_app.route("/api/cameras")
        def api_cameras() -> Response:
            return jsonify({"cameras": server._catalog(server._base_url())})

        @flask_app.route("/healthz")
        def healthz() -> Response:
            return jsonify({"ok": True, "cameras": len(server.cameras.device_ids())})

        @flask_app.route("/snapshot/<cam_id>")
        def snapshot(cam_id: str) -> Response:
            if cam_id not in server.cameras.device_ids():
                abort(404)
            frame = server.cameras.get_frame(cam_id)
            if frame is None:
                abort(503)
            jpeg = _encode_jpeg(frame, server.quality, server.max_width)
            if jpeg is None:
                abort(503)
            return Response(jpeg, mimetype="image/jpeg")

        @flask_app.route("/stream/<cam_id>")
        def stream(cam_id: str) -> Response:
            if cam_id not in server.cameras.device_ids():
                abort(404)

            def gen() -> Iterator[bytes]:
                boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                period = 1.0 / max(1.0, server.fps)
                while True:
                    t0 = time.monotonic()
                    frame = server.cameras.get_frame(cam_id)
                    if frame is not None:
                        jpeg = _encode_jpeg(frame, server.quality, server.max_width)
                        if jpeg is not None:
                            yield boundary + jpeg + b"\r\n"
                    sleep = period - (time.monotonic() - t0)
                    if sleep > 0:
                        time.sleep(sleep)

            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        return flask_app

    def start(self) -> bool:
        if not FLASK_AVAILABLE:
            _LOGGER.error("Flask not installed; cannot start stream server.")
            return False
        self._flask = self._build()

        def _serve() -> None:
            from werkzeug.serving import make_server

            backoff = 1.0
            while not self._stop:
                try:
                    self._server = make_server(self.host, self.port, self._flask, threaded=True)
                    backoff = 1.0
                    self._server.serve_forever()
                except Exception:
                    if self._stop:
                        break
                    _LOGGER.exception(
                        "Stream server accept loop crashed; restarting in %.0fs", backoff
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
                else:
                    break

        self._thread = threading.Thread(target=_serve, name="stream", daemon=True)
        self._thread.start()
        ip = lan_ip()
        _LOGGER.info("Camera streams live on http://%s:%d", ip, self.port)
        for cam_id in self.cameras.device_ids():
            _LOGGER.info("  %-24s http://%s:%d/stream/%s", cam_id, ip, self.port, cam_id)
        return True

    def stop(self) -> None:
        self._stop = True
        server = self._server
        if server is not None:
            with contextlib.suppress(Exception):
                server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
