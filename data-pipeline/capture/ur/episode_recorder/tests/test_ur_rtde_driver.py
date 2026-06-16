"""Tests for the UR + RTDE state driver (no hardware)."""

from __future__ import annotations

from episode_recorder.drivers import ur_rtde
from episode_recorder.drivers.ur_rtde import UR_JOINT_NAMES, UrRtdeDriver


class _FakeIo:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def setToolDigitalOut(self, n: int, value: bool) -> None:
        self.calls.append((n, value))


class TestUrJointNames:
    def test_six_joints(self) -> None:
        assert len(UR_JOINT_NAMES) == 6

    def test_known_names(self) -> None:
        assert UR_JOINT_NAMES[0] == "shoulder_pan_joint"
        assert UR_JOINT_NAMES[-1] == "wrist_3_joint"


class TestConstruction:
    def test_default_joint_names(self) -> None:
        assert UrRtdeDriver(host="192.168.1.80").joint_names == UR_JOINT_NAMES

    def test_custom_joint_names(self) -> None:
        names = ["j0", "j1"]
        assert UrRtdeDriver(host="x", joint_names=names).joint_names == names

    def test_default_digital_input_map(self) -> None:
        assert UrRtdeDriver(host="x")._di_map == {16: "di0", 17: "di1"}

    def test_custom_digital_input_map(self) -> None:
        assert UrRtdeDriver(host="x", digital_input_names={5: "btn"})._di_map == {5: "btn"}

    def test_absorbs_unknown_kwargs(self) -> None:
        drv = UrRtdeDriver(host="x", nats_url="ignored", cell="c")
        assert drv.host == "x"


class TestConnectWithoutHardware:
    def test_rtde_reported_unavailable(self) -> None:
        assert ur_rtde._RTDE_AVAILABLE is False

    def test_connect_false_when_rtde_unavailable(self) -> None:
        assert UrRtdeDriver(host="192.168.1.80").connect() is False

    def test_read_state_none_when_disconnected(self) -> None:
        assert UrRtdeDriver(host="x").read_state() is None


class TestSetDigitalOutput:
    def test_false_when_io_missing(self) -> None:
        assert UrRtdeDriver(host="x").set_digital_output("do0", True) is False

    def test_drives_tool_output(self) -> None:
        drv = UrRtdeDriver(host="x")
        io = _FakeIo()
        drv._io = io
        assert drv.set_digital_output("do0", True) is True
        assert io.calls == [(0, True)]

    def test_rejects_non_do_name(self) -> None:
        drv = UrRtdeDriver(host="x")
        drv._io = _FakeIo()
        assert drv.set_digital_output("di0", True) is False

    def test_rejects_unparseable_index(self) -> None:
        drv = UrRtdeDriver(host="x")
        drv._io = _FakeIo()
        assert drv.set_digital_output("doX", True) is False
