"""camera_streamer — host Orbbec cameras as MJPEG links on the LAN.

A standalone service that opens every available Orbbec camera and re-publishes
each one as a shareable HTTP MJPEG stream, so any LAN client (browser, VLC,
OpenCV, ffmpeg) can consume the feed. A dashboard lists every camera with its
direct link.
"""

from __future__ import annotations

__version__ = "0.1.0"
