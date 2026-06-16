"""Pytest setup for leader_follower: import path + ROS/SDK stubs.

The leader_follower scripts run as a flat ROS 2 package on the edge node. These
tests exercise only the pure helpers (interpolation/alignment math, episode
framing, retention pruning), so ROS, hardware, and the LeRobot/Flask stacks are
stubbed in ``sys.modules`` before any module under test is imported. numpy is
real.

ur_rtde (``rtde_control`` / ``rtde_receive``) is intentionally NOT stubbed, so
the import guards report the robot as absent (``RTDE_AVAILABLE`` stays False).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def _stub_module(name: str, **attrs: object) -> ModuleType:
    """Register (or extend) a stub module in ``sys.modules`` with ``attrs``."""
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        sys.modules[name] = module
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _StubNode:
    """Stand-in base class for rclpy.node.Node (never instantiated in tests)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        ...


class _StubMessage:
    """Placeholder ROS message type; annotations are strings under PEP 563."""


# rclpy package + the submodules the recorder nodes import at module load.
_stub_module(
    "rclpy",
    init=lambda *a, **k: None,
    shutdown=lambda *a, **k: None,
    spin=lambda *a, **k: None,
)
_stub_module("rclpy.node", Node=_StubNode)
_stub_module(
    "rclpy.qos",
    QoSProfile=type("QoSProfile", (), {"__init__": lambda self, *a, **k: None}),
    HistoryPolicy=type("HistoryPolicy", (), {"KEEP_LAST": 1}),
    ReliabilityPolicy=type("ReliabilityPolicy", (), {"BEST_EFFORT": 1, "RELIABLE": 2}),
)
_stub_module("sensor_msgs")
_stub_module("sensor_msgs.msg", JointState=_StubMessage, Image=_StubMessage)
_stub_module("std_msgs")
_stub_module("std_msgs.msg", Bool=_StubMessage, Float64=_StubMessage, String=_StubMessage)

# Heavy/optional stacks used only by the runtime nodes (not the pure helpers
# under test); stubbed so their absence never blocks an import.
for _optional in ("cv_bridge", "flask", "lerobot"):
    _stub_module(_optional)
