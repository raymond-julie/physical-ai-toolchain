"""Pytest setup for ur_dual_recorder: import path + hardware/SDK stubs.

These tests exercise only the pure configuration, camera-selection, and
gripper-threshold logic. They run with no robot hardware, no Orbbec SDK, and no
network, so the optional native/SDK modules are stubbed as empty modules in
``sys.modules`` before any package module is imported. This forces the
"unavailable / synthetic" code paths deterministically even when a developer has
some of these packages installed locally. numpy and PyYAML are real.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# Empty stubs lack the symbols the guarded imports look for (e.g.
# ``from pyorbbecsdk import Config``), so each guard raises ImportError and the
# corresponding ``*_AVAILABLE`` flag stays False.
for _name in (
    "pyorbbecsdk",
    "rtde_receive",
    "rtde_control",
    "evdev",
    "cv2",
    "flask",
    "lerobot",
):
    sys.modules.setdefault(_name, ModuleType(_name))
