"""Stub the robot client's hardware / transport dependencies for the test suite.

The wire-format and safety-clamp logic under test is pure ``numpy``, so the heavy
hardware and transport dependencies (``zmq``, ``msgpack``, ``requests``, ``cv2``,
and the ``ur_rtde`` RTDE bindings) are replaced with lightweight stand-ins in
``sys.modules`` before the module under test is imported. This lets the suite run
with no robot, camera, policy server, or cluster. ``numpy`` is used for real
because the wire format and clamps are numeric.

``setdefault`` is used so a genuinely installed dependency (e.g. ``numpy``-adjacent
``pyzmq``) is left untouched, while the typically-absent native packages
(``cv2``, ``ur_rtde``) fall back to the stub.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


class _ZMQError(Exception):
    """Stand-in for ``zmq.error.ZMQError``."""


def _install_stubs() -> None:
    zmq_error = types.ModuleType("zmq.error")
    zmq_error.ZMQError = _ZMQError

    zmq = types.ModuleType("zmq")
    zmq.Context = MagicMock()
    zmq.REQ = 3
    zmq.RCVTIMEO = 27
    zmq.SNDTIMEO = 28
    zmq.LINGER = 17
    zmq.error = zmq_error

    requests = types.ModuleType("requests")
    requests.Session = MagicMock()
    requests.get = MagicMock()

    msgpack = types.ModuleType("msgpack")
    msgpack.packb = MagicMock(return_value=b"")
    msgpack.unpackb = MagicMock(return_value={})

    cv2 = types.ModuleType("cv2")
    cv2.imdecode = MagicMock()
    cv2.cvtColor = MagicMock()
    cv2.IMREAD_COLOR = 1
    cv2.COLOR_BGR2RGB = 4

    # ur_rtde ships the rtde_receive / rtde_control bindings the client imports
    # lazily inside DualArm; stub all three so an accidental import never reaches
    # real hardware.
    ur_rtde = types.ModuleType("ur_rtde")
    rtde_receive = types.ModuleType("rtde_receive")
    rtde_receive.RTDEReceiveInterface = MagicMock()
    rtde_control = types.ModuleType("rtde_control")
    rtde_control.RTDEControlInterface = MagicMock()

    stubs = {
        "zmq": zmq,
        "zmq.error": zmq_error,
        "requests": requests,
        "msgpack": msgpack,
        "cv2": cv2,
        "ur_rtde": ur_rtde,
        "rtde_receive": rtde_receive,
        "rtde_control": rtde_control,
    }
    for name, module in stubs.items():
        sys.modules.setdefault(name, module)


_install_stubs()
