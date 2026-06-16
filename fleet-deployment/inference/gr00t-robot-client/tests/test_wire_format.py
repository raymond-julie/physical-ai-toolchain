"""Behavior tests for the GR00T dual-arm inference client's pure logic.

Two pieces of non-trivial, hardware-free logic are covered:

* the GR00T ZMQ wire format -- the msgpack ``default`` / ``object_hook`` hooks
  that serialize numpy arrays as ``.npy`` bytes and read them back, and
* the safety clamps -- the first-pose start gate, the per-step clamp
  (``--max-joint-step``), and the absolute clamp to the trained action range from
  ``metadata.json`` (``--metadata``).

Hardware / transport dependencies are stubbed in ``conftest.py`` before this
module imports the client, so the suite needs no robot, camera, policy server, or
cluster. ``numpy`` is real because every function under test is numeric.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# gr00t-robot-client is hyphenated, so the client cannot be imported as a normal
# package. Load it by file path; conftest has already installed the stubs.
_SUT_PATH = Path(__file__).resolve().parents[1] / "robot_inference_client.py"
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

    def test_encoded_payload_is_npy_format(self):
        # The hook must emit a real .npy stream (magic prefix), matching
        # gr00t.eval.service.MsgSerializer, not some ad-hoc encoding.
        encoded = sut._encode_custom(np.zeros((2, 2), dtype=np.float32))
        assert bytes(encoded["as_npy"][:6]) == b"\x93NUMPY"

    def test_roundtrip_preserves_uint8_image(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(2, 480, 848, 3), dtype=np.uint8)
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
        with pytest.raises(ValueError):
            sut._encode_custom(np.array([object()], dtype=object))


# ---------------------------------------------------------------------------
# Per-step clamp (--max-joint-step): caps joint speed
# ---------------------------------------------------------------------------
class TestStepClamp:
    def test_limits_large_positive_motion(self):
        reference = np.zeros(6, dtype=np.float32)
        target = np.full(6, 1.0, dtype=np.float32)
        clamped = sut.clamp_step(target, reference, max_step=0.03)
        assert np.allclose(clamped, 0.03)

    def test_limits_large_negative_motion(self):
        reference = np.zeros(6, dtype=np.float32)
        target = np.full(6, -1.0, dtype=np.float32)
        clamped = sut.clamp_step(target, reference, max_step=0.03)
        assert np.allclose(clamped, -0.03)

    def test_passes_small_motion_unchanged(self):
        reference = np.zeros(6, dtype=np.float32)
        target = np.array([0.01, -0.02, 0.0, 0.005, -0.005, 0.029], dtype=np.float32)
        clamped = sut.clamp_step(target, reference, max_step=0.03)
        assert np.allclose(clamped, target)

    def test_clamp_is_per_joint(self):
        reference = np.zeros(6, dtype=np.float32)
        target = np.array([1.0, 0.01, -1.0, -0.02, 0.5, 0.0], dtype=np.float32)
        clamped = sut.clamp_step(target, reference, max_step=0.03)
        assert np.allclose(clamped, [0.03, 0.01, -0.03, -0.02, 0.03, 0.0])

    def test_clamp_is_relative_to_reference(self):
        reference = np.full(6, 0.5, dtype=np.float32)
        target = np.full(6, 0.9, dtype=np.float32)
        clamped = sut.clamp_step(target, reference, max_step=0.03)
        assert np.allclose(clamped, 0.53)


# ---------------------------------------------------------------------------
# Absolute clamp (--metadata): clip to the trained action range plus margin
# ---------------------------------------------------------------------------
class TestAbsoluteClamp:
    def test_clips_above_and_below_bounds(self):
        lo = np.zeros(6, dtype=np.float32)
        hi = np.ones(6, dtype=np.float32)
        target = np.array([-1.0, 2.0, 0.5, 1.5, -0.5, 0.25], dtype=np.float32)
        clamped = sut.clamp_absolute(target, (lo, hi), margin=0.1)
        # bounds widened by margin: [-0.1, 1.1].
        assert np.allclose(clamped, [-0.1, 1.1, 0.5, 1.1, -0.1, 0.25])

    def test_no_bounds_is_passthrough(self):
        target = np.array([5.0, -5.0, 0.0], dtype=np.float32)
        assert np.array_equal(sut.clamp_absolute(target, None, margin=0.1), target)

    def test_within_bounds_unchanged(self):
        lo = np.full(6, -2.0, dtype=np.float32)
        hi = np.full(6, 2.0, dtype=np.float32)
        target = np.array([0.0, 1.0, -1.0, 1.9, -1.9, 0.5], dtype=np.float32)
        assert np.allclose(sut.clamp_absolute(target, (lo, hi), margin=0.0), target)

    def test_load_action_bounds_reads_min_max(self, tmp_path):
        meta = {
            "new_embodiment": {
                "statistics": {
                    "action": {
                        "robot1_arm": {
                            "min": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                            "max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        },
                        "robot2_arm": {
                            "min": [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                            "max": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                        },
                    }
                }
            }
        }
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(meta), encoding="utf-8")

        bounds = sut.load_action_bounds(str(path))

        assert bounds is not None
        assert set(bounds) == {"robot1_arm", "robot2_arm"}
        lo1, hi1 = bounds["robot1_arm"]
        assert lo1.dtype == np.float32 and hi1.dtype == np.float32
        assert np.allclose(lo1, -1.0) and np.allclose(hi1, 1.0)
        assert np.allclose(bounds["robot2_arm"][0], -2.0)
        assert np.allclose(bounds["robot2_arm"][1], 2.0)

    def test_load_bounds_then_clamp_uses_trained_range(self, tmp_path):
        meta = {
            "new_embodiment": {
                "statistics": {
                    "action": {
                        "robot1_arm": {"min": [0.0] * 6, "max": [1.0] * 6},
                        "robot2_arm": {"min": [0.0] * 6, "max": [1.0] * 6},
                    }
                }
            }
        }
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(meta), encoding="utf-8")
        bounds = sut.load_action_bounds(str(path))

        target = np.full(6, 5.0, dtype=np.float32)
        clamped = sut.clamp_absolute(target, bounds["robot1_arm"], margin=0.0)
        assert np.allclose(clamped, 1.0)

    def test_load_action_bounds_missing_file_returns_none(self, tmp_path):
        assert sut.load_action_bounds(str(tmp_path / "does-not-exist.json")) is None


# ---------------------------------------------------------------------------
# First-pose start gate: refuse to start when the policy would jump the arms
# ---------------------------------------------------------------------------
class TestFirstPoseGate:
    def test_jump_is_max_abs_per_joint_across_arms(self):
        q1 = np.zeros(6, dtype=np.float32)
        q2 = np.zeros(6, dtype=np.float32)
        a1 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        a2 = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float32)
        assert sut.first_pose_jump(a1, a2, q1, q2) == pytest.approx(0.4)

    def test_jump_is_zero_when_pose_matches(self):
        q1 = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
        q2 = np.linspace(0.0, 0.5, 6, dtype=np.float32)
        assert sut.first_pose_jump(q1.copy(), q2.copy(), q1, q2) == pytest.approx(0.0)

    def test_gate_triggers_above_threshold(self):
        q = np.zeros(6, dtype=np.float32)
        a1 = np.full(6, 0.5, dtype=np.float32)
        jump = sut.first_pose_jump(a1, q, q, q)
        # Mirrors Controller.run: jump > start_threshold and not allow_jump.
        assert jump > 0.30

    def test_gate_allows_below_threshold(self):
        q = np.zeros(6, dtype=np.float32)
        a1 = np.full(6, 0.1, dtype=np.float32)
        jump = sut.first_pose_jump(a1, q, q, q)
        assert jump <= 0.30
