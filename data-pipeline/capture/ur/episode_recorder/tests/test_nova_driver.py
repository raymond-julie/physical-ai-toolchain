"""Tests for the Nova driver observation-assembly (``read_state``) logic.

These bypass NATS entirely: the driver is constructed (no ``connect``),
its latest-payload slot is populated directly, and ``read_state`` is
exercised — so no network or ``nats-py`` install is required.
"""

from __future__ import annotations

from typing import Any

from episode_recorder.drivers.nova import NovaDriver
from episode_recorder.drivers.ur_rtde import UR_JOINT_NAMES


def _driver(**kwargs: Any) -> NovaDriver:
    return NovaDriver(**kwargs)


def _payload(joint_position: list[Any]) -> dict:
    return {"motion_groups": [{"joint_position": joint_position}]}


class TestReadStateNoData:
    def test_none_before_first_message(self) -> None:
        assert _driver().read_state() is None

    def test_empty_motion_groups(self) -> None:
        drv = _driver()
        drv._latest = {"motion_groups": []}
        assert drv.read_state() is None

    def test_missing_joint_position(self) -> None:
        drv = _driver()
        drv._latest = {"motion_groups": [{}]}
        assert drv.read_state() is None


class TestReadStateAssembly:
    def test_matching_joint_count(self) -> None:
        drv = _driver()
        drv._latest = _payload([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        state = drv.read_state()
        assert state is not None
        assert state.joint_names == UR_JOINT_NAMES
        assert state.joint_positions == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def test_velocities_are_zeroed(self) -> None:
        drv = _driver()
        drv._latest = _payload([0.0] * 6)
        state = drv.read_state()
        assert state is not None
        assert state.joint_velocities == [0.0] * 6

    def test_digital_inputs_empty(self) -> None:
        drv = _driver()
        drv._latest = _payload([0.0] * 6)
        state = drv.read_state()
        assert state is not None
        assert state.digital_inputs == {}

    def test_positions_coerced_to_float(self) -> None:
        drv = _driver()
        drv._latest = _payload(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
        state = drv.read_state()
        assert state is not None
        assert state.joint_positions == [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert all(isinstance(p, float) for p in state.joint_positions)

    def test_truncates_names_when_fewer_joints(self) -> None:
        drv = _driver()
        drv._latest = _payload([0.0, 0.1, 0.2])
        state = drv.read_state()
        assert state is not None
        assert state.joint_names == UR_JOINT_NAMES[:3]
        assert len(state.joint_positions) == 3

    def test_pads_names_when_more_joints(self) -> None:
        drv = _driver()
        drv._latest = _payload([0.0] * 8)
        state = drv.read_state()
        assert state is not None
        assert state.joint_names == [*UR_JOINT_NAMES, "joint_6", "joint_7"]
        assert len(state.joint_velocities) == 8

    def test_custom_joint_names(self) -> None:
        names = ["a", "b", "c"]
        drv = _driver(joint_names=names)
        drv._latest = _payload([1.0, 2.0, 3.0])
        state = drv.read_state()
        assert state is not None
        assert state.joint_names == names


class TestConstruction:
    def test_subject_template_filled(self) -> None:
        drv = _driver(cell="mycell", controller="ur5-left")
        assert drv.subject == "nova.v2.cells.mycell.controllers.ur5-left.state"

    def test_absorbs_unknown_kwargs(self) -> None:
        drv = _driver(host="192.168.1.80", port=0, robot_driver="nova")
        assert drv.cell == "cell"
