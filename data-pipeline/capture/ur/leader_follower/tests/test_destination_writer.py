"""Tests for the destination-writer pure helpers (no ROS, no hardware).

Covers the alignment/interpolation math the state machine relies on:
``clamp_to_limits``, ``interpolate_toward`` (in-place stepping toward a target),
and ``at_home`` (home-pose proximity).
"""

from __future__ import annotations

import math

from destination_writer import at_home, clamp_to_limits, interpolate_toward


class TestClampToLimits:
    def test_value_within_limits_is_unchanged(self) -> None:
        assert clamp_to_limits(0.5, (-1.0, 1.0)) == 0.5

    def test_value_above_upper_limit_is_clamped(self) -> None:
        assert clamp_to_limits(2.0, (-1.0, 1.0)) == 1.0

    def test_value_below_lower_limit_is_clamped(self) -> None:
        assert clamp_to_limits(-5.0, (-1.0, 1.0)) == -1.0

    def test_value_at_bounds_is_preserved(self) -> None:
        assert clamp_to_limits(1.0, (-1.0, 1.0)) == 1.0
        assert clamp_to_limits(-1.0, (-1.0, 1.0)) == -1.0


class TestInterpolateToward:
    def test_within_one_step_snaps_to_target(self) -> None:
        current = [0.0, 0.0]
        worst, excess = interpolate_toward(current, [0.1, -0.2], max_step=0.5)
        assert current == [0.1, -0.2]
        assert worst == 0.2
        assert excess == 0.0

    def test_positive_error_steps_by_max_step(self) -> None:
        current = [0.0]
        worst, excess = interpolate_toward(current, [1.0], max_step=0.3)
        assert math.isclose(current[0], 0.3)
        assert worst == 1.0
        assert excess == 1.0

    def test_negative_error_steps_downward(self) -> None:
        current = [0.0]
        _worst, _excess = interpolate_toward(current, [-1.0], max_step=0.25)
        assert math.isclose(current[0], -0.25)

    def test_mixed_joints_step_independently(self) -> None:
        current = [0.0, 0.0]
        worst, excess = interpolate_toward(current, [0.1, 2.0], max_step=0.5)
        assert math.isclose(current[0], 0.1)  # within one step -> snapped
        assert math.isclose(current[1], 0.5)  # exceeds -> stepped
        assert worst == 2.0
        assert excess == 2.0


class TestAtHome:
    HOME = [0.0, 1.0, -1.0]

    def test_none_servo_positions_is_not_home(self) -> None:
        assert at_home(None, self.HOME, threshold=0.1) is False

    def test_exact_match_is_home(self) -> None:
        assert at_home(list(self.HOME), self.HOME, threshold=0.01) is True

    def test_within_threshold_is_home(self) -> None:
        servo = [0.005, 1.005, -1.005]
        assert at_home(servo, self.HOME, threshold=0.01) is True

    def test_one_joint_outside_threshold_is_not_home(self) -> None:
        servo = [0.0, 1.0, -0.5]
        assert at_home(servo, self.HOME, threshold=0.1) is False
