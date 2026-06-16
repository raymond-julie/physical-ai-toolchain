"""Flask web dashboard for ur_dual_recorder.

Provides:

* ``GET  /``                  — dashboard page.
* ``GET  /api/status``        — JSON status of arms / recording / cameras.
* ``POST /api/record``        — toggle episode recording.
* ``POST /api/defer_encoding`` — hold stopped episodes raw until requested.
* ``POST /api/encode_pending`` — flush held raw episodes into the encoder.
* ``GET  /api/preview/<cam>`` — MJPEG live preview of a camera.

The server runs in a background thread so the main process keeps ownership of the
control loop.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from flask import Flask

    from ..app import DualRecorderApp

_LOGGER = logging.getLogger(__name__)

try:
    from flask import Flask, Response, jsonify, render_template, request

    FLASK_AVAILABLE = True
except ImportError:  # pragma: no cover
    FLASK_AVAILABLE = False


def _encode_jpeg(frame: np.ndarray) -> bytes | None:
    try:
        import cv2

        # Downscale previews so 4 simultaneous MJPEG streams don't burn CPU on
        # full-res JPEG encoding (the cameras record at full res regardless).
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(
                frame, (640, round(h * scale)), interpolation=cv2.INTER_AREA
            )
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return buf.tobytes() if ok else None
    except Exception:
        return None


class WebServer:
    """Background Flask server bound to a :class:`DualRecorderApp`."""

    def __init__(
        self, app: DualRecorderApp, host: str = "0.0.0.0", port: int = 8080
    ) -> None:
        self.app = app
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None
        self._flask: Flask | None = None
        self._server: Any = None
        self._stop = False

    def _build(self) -> Flask:
        flask_app = Flask(
            __name__,
            template_folder="templates",
            static_folder="static",
        )
        recorder_app = self.app

        @flask_app.route("/")
        def index() -> str:
            return render_template("index.html")

        @flask_app.route("/api/status")
        def status() -> Any:
            return jsonify(recorder_app.status())

        @flask_app.route("/api/record", methods=["POST"])
        def record() -> Any:
            recorder_app.toggle_recording()
            return jsonify({"recording": recorder_app.is_recording})

        @flask_app.route("/api/defer_encoding", methods=["POST"])
        def defer_encoding() -> Any:
            data = request.get_json(silent=True) or {}
            enabled = bool(data.get("enabled", False))
            recorder_app.set_defer_encoding(enabled)
            return jsonify({"defer_encoding": enabled})

        @flask_app.route("/api/encode_pending", methods=["POST"])
        def encode_pending() -> Any:
            queued = recorder_app.encode_pending()
            return jsonify({"queued": queued})

        @flask_app.route("/api/preview/<cam_id>")
        def preview(cam_id: str) -> Any:
            def gen() -> Iterator[bytes]:
                boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                while True:
                    frame = recorder_app.cameras.get_frame(cam_id)
                    if frame is not None:
                        jpeg = _encode_jpeg(frame)
                        if jpeg is not None:
                            yield boundary + jpeg + b"\r\n"
                    time.sleep(1 / 10.0)

            return Response(
                gen(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        return flask_app

    def start(self) -> bool:
        if not FLASK_AVAILABLE:
            _LOGGER.warning("Flask not installed; web GUI disabled.")
            return False
        self._flask = self._build()

        def _serve() -> None:
            from werkzeug.serving import make_server

            backoff = 1.0
            while not self._stop:
                try:
                    # make_server sets SO_REUSEADDR, so a rebind after a crash
                    # succeeds immediately. Running it ourselves (instead of
                    # flask.run) lets us catch and log a dying accept loop and
                    # restart it, rather than the daemon thread vanishing silently
                    # under load (e.g. during episode encoding).
                    self._server = make_server(
                        self.host, self.port, self._flask, threaded=True
                    )
                    backoff = 1.0
                    self._server.serve_forever()
                except Exception:
                    if self._stop:
                        break
                    _LOGGER.exception(
                        "Web server accept loop crashed; restarting in %.0fs", backoff
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
                else:
                    # serve_forever returned normally => intentional shutdown.
                    break

        self._thread = threading.Thread(target=_serve, name="web", daemon=True)
        self._thread.start()
        _LOGGER.info("Web GUI on http://%s:%d", self.host, self.port)
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
