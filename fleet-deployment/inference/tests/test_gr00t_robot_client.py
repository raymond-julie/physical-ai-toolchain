"""Behavior tests for the GR00T dual-arm inference client.

Covers the two pieces of non-trivial pure logic in `robot_inference_client`:

* the GR00T ZMQ wire format (msgpack ``default`` / ``object_hook`` hooks that
  serialize numpy arrays as ``.npy`` bytes), and
* the safety clamps (first-pose gate, per-step clamp, absolute metadata clamp).

External hardware/cluster dependencies (`zmq`, `requests`, `msgpack`) are stubbed
before the module under test is imported, so the suite needs no robot, camera,
policy server, or cluster. `numpy` is used for real because the wire format and
clamps are numeric.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub the hardware / network dependencies BEFORE importing the SUT. Only
# module importability is required here; the pure functions under test use
# real numpy and never touch these stubs.
# ---------------------------------------------------------------------------


class _ZMQError(Exception):
    """Stand-in for zmq.error.ZMQError."""


_zmq_error = types.ModuleType("zmq.error")
_zmq_error.ZMQError = _ZMQError

_zmq = types.ModuleType("zmq")
_zmq.Context = MagicMock()
_zmq.REQ = 3
_zmq.RCVTIMEO = 27
_zmq.SNDTIMEO = 28
_zmq.LINGER = 17
_zmq.error = _zmq_error

_requests = types.ModuleType("requests")
_requests.Session = MagicMock()
_requests.get = MagicMock()

_msgpack = types.ModuleType("msgpack")
_msgpack.packb = MagicMock(return_value=b"")
_msgpack.unpackb = MagicMock(return_value={})

sys.modules["zmq"] = _zmq
sys.modules["zmq.error"] = _zmq_error
sys.modules["requests"] = _requests
sys.modules["msgpack"] = _msgpack

# fleet-deployment is hyphenated so the client cannot be imported as a normal
# package. Load it by file path after the stubs are installed.
_SUT_PATH = Path(__file__).resolve().parent.parent / "gr00t-robot-client" / "robot_inference_client.py"
_spec = importlib.util.spec_from_file_location("robot_inference_client", _SUT_PATH)
sut = importlib.util.module_from_spec(_spec)
sys.modules["robot_inference_client"] = sut
_spec.loader.exec_module(sut)


# ---------------------------------------------------------------------------
# Wire format: msgpack default / object_hook ndarray serialization
# ---------------------------------------------------------------------------
class TestWireFormat:
    def test_encode_roundtrips_through_decode(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        encoded = sut._encode_custom(arr)
        assert encoded["__ndarray_class__"] is True
        assert isinstance(encoded["as_npy"], (bytes, bytearray))
        decoded = sut._decode_custom(encoded)
        assert np.array_equal(decoded, arr)
        assert decoded.dtype == arr.dtype
        assert decoded.shape == arr.shape

    def test_roundtrip_preserves_uint8_image(self):
        img = np.random.randint(0, 256, size=(2, 480, 848, 3), dtype=np.uint8)
        decoded = sut._decode_custom(sut._encode_custom(img))
        assert decoded.dtype == np.uint8
        assert np.array_equal(decoded, img)

    def test_encode_rejects_non_ndarray(self):
        with pytest.raises(TypeError):
            sut._encode_custom(object())

    def test_decode_passes_through_plain_dict(self):
        payload = {"endpoint": "ping", "value": 1}
        assert sut._decode_custom(payload) == payload

    def test_encode_disallows_pickle(self):
        # Object arrays cannot be saved with allow_pickle=False; the hook must
        # refuse rather than silently emit a pickled payload.
        obj_array = np.array([object()], dtype=object)
        with pytest.raises(ValueError):
            sut._encode_custom(obj_array)


# ---------------------------------------------------------------------------
# Observation assembly
# ---------------------------------------------------------------------------
class TestBuildObservation:
    def test_shapes_and_dtypes(self):
        frames = [np.zeros((480, 848, 3), dtype=np.uint8) for _ in range(4)]
        q1 = np.zeros(sut.ARM_DOF, dtype=np.float32)
        q2 = np.ones(sut.ARM_DOF, dtype=np.float32)
        obs = sut.build_observation(frames, q1, q2, "pick up the block")

        for idx in range(4):
            key = f"video.color_{idx}"
            assert obs[key].shape == (1, 480, 848, 3)
            assert obs[key].dtype == np.uint8
        assert obs["state.robot1_arm"].shape == (1, sut.ARM_DOF)
        assert obs["state.robot2_arm"].shape == (1, sut.ARM_DOF)
        assert obs["state.robot1_arm"].dtype == np.float32
        assert obs["annotation.human.action.task_description"] == ["pick up the block"]


# ---------------------------------------------------------------------------
# Per-step clamp
# ---------------------------------------------------------------------------
class TestClampStep:
    def test_caps_large_moves_to_max_step(self):
        ref = np.zeros(6, dtype=np.float32)
        target = np.array([1.0, -1.0, 0.01, 0.0, 0.5, -0.5], dtype=np.float32)
        out = sut.clamp_step(target, ref, max_step=0.03)
        assert np.all(np.abs(out - ref) <= 0.03 + 1e-6)
        # A small move passes through unchanged.
        assert out[2] == pytest.approx(0.01)
        # Large moves saturate to +/- max_step.
        assert out[0] == pytest.approx(0.03)
        assert out[1] == pytest.approx(-0.03)

    def test_clamp_is_relative_to_reference(self):
        ref = np.full(6, 1.0, dtype=np.float32)
        target = np.full(6, 2.0, dtype=np.float32)
        out = sut.clamp_step(target, ref, max_step=0.1)
        assert np.allclose(out, 1.1)


# ---------------------------------------------------------------------------
# Absolute metadata clamp
# ---------------------------------------------------------------------------
class TestClampAbsolute:
    def test_clamps_to_bounds_plus_margin(self):
        lo = np.zeros(6, dtype=np.float32)
        hi = np.ones(6, dtype=np.float32)
        target = np.array([-1.0, 2.0, 0.5, 0.0, 1.0, 0.9], dtype=np.float32)
        out = sut.clamp_absolute(target, (lo, hi), margin=0.1)
        assert out[0] == pytest.approx(-0.1)  # below lo -> lo - margin
        assert out[1] == pytest.approx(1.1)  # above hi -> hi + margin
        assert out[2] == pytest.approx(0.5)  # inside range, untouched

    def test_none_bounds_pass_through(self):
        target = np.array([5.0, -5.0], dtype=np.float32)
        out = sut.clamp_absolute(target, None, margin=0.1)
        assert np.array_equal(out, target)


# ---------------------------------------------------------------------------
# Metadata bounds loading
# ---------------------------------------------------------------------------
class TestLoadActionBounds:
    @staticmethod
    def _write_metadata(tmp_path: Path) -> Path:
        meta = {
            "new_embodiment": {
                "statistics": {
                    "action": {
                        "robot1_arm": {"min": [0.0] * 6, "max": [1.0] * 6},
                        "robot2_arm": {"min": [-1.0] * 6, "max": [0.0] * 6},
                    }
                }
            }
        }
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(meta))
        return path

    def test_loads_both_arm_bounds(self, tmp_path):
        bounds = sut.load_action_bounds(str(self._write_metadata(tmp_path)))
        assert set(bounds) == {"robot1_arm", "robot2_arm"}
        lo, hi = bounds["robot1_arm"]
        assert np.allclose(lo, 0.0)
        assert np.allclose(hi, 1.0)
        assert lo.dtype == np.float32

    def test_missing_file_returns_none(self, tmp_path):
        assert sut.load_action_bounds(str(tmp_path / "absent.json")) is None

    def test_malformed_json_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert sut.load_action_bounds(str(path)) is None


# ---------------------------------------------------------------------------
# First-pose gate
# ---------------------------------------------------------------------------
class TestFirstPoseGate:
    def test_returns_max_per_joint_distance_across_both_arms(self):
        q = np.zeros(6, dtype=np.float32)
        a1 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        a2 = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert sut.first_pose_jump(a1, a2, q, q) == pytest.approx(0.5)

    def test_gate_allows_pose_within_threshold(self):
        q = np.zeros(6, dtype=np.float32)
        near = np.full(6, 0.1, dtype=np.float32)
        jump = sut.first_pose_jump(near, near, q, q)
        assert jump <= 0.30

    def test_gate_refuses_pose_beyond_threshold(self):
        q = np.zeros(6, dtype=np.float32)
        far = np.full(6, 0.5, dtype=np.float32)
        jump = sut.first_pose_jump(far, far, q, q)
        assert jump > 0.30
