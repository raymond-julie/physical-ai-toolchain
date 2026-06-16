"""Pytest setup for episode_recorder: import path + hardware/SDK stubs.

These tests exercise only the pure driver-registry and observation
assembly logic. They run with no ROS install, no robot hardware, and no
network, so the optional native/SDK modules are stubbed in
``sys.modules`` before any driver module is imported.

We deliberately do NOT stub ``rtde_receive`` / ``rtde_io`` (the modules
the ur_rtde driver probes), so :class:`UrRtdeDriver` reports the robot
as absent (``_RTDE_AVAILABLE`` stays False) — its read paths are then
covered via monkeypatching rather than live RTDE calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# Stub optional native / SDK dependencies so importing any driver (or a
# node, for future tests) needs no ROS / hardware / network.
for _name in ("rclpy", "ur_rtde", "nats", "aiortc", "av", "cv2"):
    sys.modules.setdefault(_name, ModuleType(_name))
