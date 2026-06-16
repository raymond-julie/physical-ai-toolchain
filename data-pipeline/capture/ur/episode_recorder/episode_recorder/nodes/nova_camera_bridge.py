"""Nova WebRTC -> ROS 2 ``sensor_msgs/Image`` bridge.

Replaces the local ``realsense2_camera`` launch when frames are sourced
from Wandelbots Nova's RealSense camera-manager app instead of from
RealSense USB devices attached directly to this host.

Architecture
------------
Nova exposes a FastAPI service per cell (e.g.
``http://<nova-host>/cell/realsense``) that:

  * lists connected cameras under ``GET /api/devices/``;
  * accepts a WebRTC handshake under ``POST /api/webrtc/{offer,answer}``
    and streams the requested per-camera tracks (color / depth) over
    a single ``RTCPeerConnection``.

The "server-offer" flow is used: we ``POST /api/webrtc/offer`` with
``{device_id, stream_types}`` and ``sdp`` omitted, the server returns
its SDP offer, we answer with our local SDP via
``POST /api/webrtc/answer``. ICE candidates are inlined in both SDPs
(non-trickle), so no further round-trip is needed.

Incoming :class:`av.VideoFrame` instances are converted to ``rgb24``
``numpy.ndarray`` and published as ``sensor_msgs/Image`` on the same
topic name that the local realsense launcher uses
(``/<camera_name>/<camera_name>/color/image_raw``), so the recorder
and GUI nodes consume Nova-sourced frames transparently.

Parameters
----------
``api_base``       Nova RealSense app URL, e.g.
                   ``http://192.168.1.71/cell/realsense``.
``device_id``      Camera serial number (Nova device id).
``camera_name``    ROS namespace + topic prefix (e.g. ``camera1``).
``stream_types``   List of Nova stream kinds to request. Defaults to
                   ``["color"]``.
``frame_id``       Optional ``header.frame_id`` (defaults to
                   ``camera_name``).
``reconnect_s``    Backoff between reconnect attempts.

This node intentionally has no aiortc/av imports at module top so that
``import episode_recorder.nodes.nova_camera_bridge`` does not fail in
environments where the deps are missing — the failure surfaces at
``main()`` with a clear message instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# Force STRING typing for parameters whose values are all-digit serial
# numbers (e.g. "353322270772"). Without this descriptor, the ROS 2
# CLI auto-detects integer overrides and ``declare_parameter`` rejects
# the override with ``InvalidParameterTypeException``.
_STR_DESC = ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
_STR_ARR_DESC = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)


def _http_json(
    method: str, url: str, payload: dict | None = None, timeout: float = 10.0
) -> dict:
    """Synchronous JSON HTTP helper (stdlib only — keeps deps minimal)."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


class NovaCameraBridge(Node):
    """One bridge node per Nova camera serial."""

    def __init__(self) -> None:
        super().__init__("nova_camera_bridge")

        # ── Parameters ──────────────────────────────────────────
        self.declare_parameter("api_base", "http://192.168.1.71/cell/realsense", _STR_DESC)
        self.declare_parameter("device_id", "", _STR_DESC)
        self.declare_parameter("camera_name", "camera1", _STR_DESC)
        self.declare_parameter("stream_types", ["color"], _STR_ARR_DESC)
        self.declare_parameter("frame_id", "", _STR_DESC)
        self.declare_parameter("reconnect_s", 5.0)
        self.declare_parameter("publish_qos_depth", 5)

        self.api_base = str(self.get_parameter("api_base").value).rstrip("/")
        self.device_id = str(self.get_parameter("device_id").value).strip()
        self.camera_name = str(self.get_parameter("camera_name").value).strip() or "camera1"
        raw_types = self.get_parameter("stream_types").value or ["color"]
        self.stream_types: list[str] = [str(t).strip() for t in raw_types if str(t).strip()] or [
            "color"
        ]
        self.frame_id = str(self.get_parameter("frame_id").value).strip() or self.camera_name
        self.reconnect_s = float(self.get_parameter("reconnect_s").value)

        if not self.device_id:
            self.get_logger().fatal('nova_camera_bridge: parameter "device_id" is required')
            raise SystemExit(2)

        # Match the topic shape of the local realsense2_camera launch
        # (camera_namespace=<name>, camera_name=<name>) so the recorder
        # + GUI configuration is unchanged.
        topic = f"/{self.camera_name}/{self.camera_name}/color/image_raw"
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.get_parameter("publish_qos_depth").value),
        )
        self.pub = self.create_publisher(Image, topic, qos)
        self._topic = topic

        self._frames_published = 0
        self._last_log_t = time.monotonic()

        # Stop signal for the asyncio worker.
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"NovaCameraBridge[{self.camera_name}]", daemon=True
        )
        self._thread.start()

        self.get_logger().info(
            f"nova_camera_bridge[{self.camera_name}]: api={self.api_base} "
            f"device_id={self.device_id} stream_types={self.stream_types} "
            f"-> {topic}"
        )

    # ── Lifecycle ──────────────────────────────────────────────

    def destroy_node(self) -> bool:  # type: ignore[override]
        self._stop.set()
        with contextlib.suppress(Exception):
            self._thread.join(timeout=2.0)
        return super().destroy_node()

    # ── asyncio worker ─────────────────────────────────────────

    def _run(self) -> None:
        try:
            import av  # noqa: F401
            from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: F401
        except ImportError as e:
            self.get_logger().fatal(
                f"aiortc / av not installed ({e}). Install with: pip install --user aiortc av"
            )
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._supervisor())
        except Exception as e:
            self.get_logger().error(f"webrtc supervisor crashed: {e}")
        finally:
            with contextlib.suppress(Exception):
                loop.close()

    async def _supervisor(self) -> None:
        """Reconnect forever until ``_stop`` is set."""
        while not self._stop.is_set() and rclpy.ok():
            try:
                await self._session_once()
            except Exception as e:
                self.get_logger().warning(
                    f"webrtc session failed: {e!r}; reconnecting in {self.reconnect_s:.1f}s"
                )
            else:
                self.get_logger().info(
                    f"webrtc session ended; reconnecting in {self.reconnect_s:.1f}s"
                )
            # Sleep in small slices so destroy_node returns quickly.
            for _ in range(int(self.reconnect_s * 10)):
                if self._stop.is_set() or not rclpy.ok():
                    return
                await asyncio.sleep(0.1)

    async def _session_once(self) -> None:
        from aiortc import RTCPeerConnection, RTCSessionDescription

        pc = RTCPeerConnection()
        # NOTE: with the server-offer flow, transceivers are created
        # by ``setRemoteDescription`` from the offer SDP — we must
        # NOT pre-add recvonly transceivers here, otherwise aiortc
        # raises ``InvalidAccessError: Media section count mismatch``.

        consumers: list[asyncio.Task] = []

        @pc.on("track")
        def _on_track(track: Any) -> None:
            self.get_logger().info(f"webrtc track received: kind={track.kind} id={track.id}")
            if track.kind == "video":
                consumers.append(asyncio.ensure_future(self._consume_track(track)))

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            self.get_logger().info(f"webrtc connection state: {pc.connectionState}")

        # ── server-offer handshake ─────────────────────────────
        offer = _http_json(
            "POST",
            f"{self.api_base}/api/webrtc/offer",
            payload={
                "device_id": self.device_id,
                "stream_types": list(self.stream_types),
            },
        )
        session_id = offer.get("session_id")
        sdp = offer.get("sdp")
        if not session_id or not sdp:
            raise RuntimeError(f"unexpected /webrtc/offer response: {offer!r}")

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        _http_json(
            "POST",
            f"{self.api_base}/api/webrtc/answer",
            payload={
                "session_id": session_id,
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            },
        )
        self.get_logger().info(f"webrtc session {session_id} established")

        try:
            # Run until either side closes or stop is requested.
            while not self._stop.is_set() and rclpy.ok():
                if pc.connectionState in ("failed", "closed"):
                    break
                await asyncio.sleep(0.2)
        finally:
            for t in consumers:
                t.cancel()
            with contextlib.suppress(Exception):
                await pc.close()
            # Best-effort server-side cleanup.
            with contextlib.suppress(urllib.error.URLError, urllib.error.HTTPError):
                _http_json("DELETE", f"{self.api_base}/api/webrtc/sessions/{session_id}")

    async def _consume_track(self, track: Any) -> None:
        """Read frames from the WebRTC track and publish to ROS."""
        while not self._stop.is_set() and rclpy.ok():
            try:
                frame = await track.recv()
            except Exception as e:  # MediaStreamError etc.
                self.get_logger().info(f"webrtc track ended: {e!r}")
                return
            try:
                self._publish_frame(frame)
            except Exception as e:
                self.get_logger().warning(f"frame publish failed: {e!r}")

    def _publish_frame(self, frame: Any) -> None:
        # av.VideoFrame -> contiguous RGB ndarray (H, W, 3)
        img = frame.to_ndarray(format="rgb24")
        if img.ndim != 3 or img.shape[2] != 3:
            return
        h, w = img.shape[:2]
        msg = Image()
        # Stamp with the ROS clock; aiortc's frame.pts is in 90 kHz
        # units relative to an opaque epoch, not wall-clock, so it
        # would only confuse downstream consumers.
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(w * 3)
        # tobytes() is required: rclpy needs `bytes`, not a numpy array.
        msg.data = img.tobytes()
        self.pub.publish(msg)

        self._frames_published += 1
        now = time.monotonic()
        if now - self._last_log_t >= 5.0:
            dt = now - self._last_log_t
            fps = self._frames_published / dt if dt > 0 else 0.0
            self.get_logger().info(
                f"{self.camera_name}: {self._frames_published} frames in "
                f"{dt:.1f}s ({fps:.1f} fps) -> {self._topic}"
            )
            self._frames_published = 0
            self._last_log_t = now


def main() -> None:
    rclpy.init()
    node: NovaCameraBridge | None = None
    try:
        node = NovaCameraBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
