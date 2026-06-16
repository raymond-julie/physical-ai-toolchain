"""ur_dual_recorder — dual follower-arm UR episode recorder (no ROS).

A standalone (no-ROS) application that:

* loads its robot/camera topology from ``/etc/trainmybot/config_v3.yaml``,
* reads two follower UR arms' joint positions plus each Robotiq 2F-85 gripper
  state over RTDE (read-only; never takes the control lock),
* captures every configured Orbbec camera,
* records synchronized episodes as a LeRobotDataset (parquet + mp4).

The package is intentionally framework-light: every subsystem is a plain Python
object so it can be unit-tested without robot hardware present. The shelved
teleoperation modules (``analog``, ``ur_interface``, ``teleop``) remain in the
tree for when mirroring is re-enabled, but are not imported in recording-only
mode.
"""

from __future__ import annotations

__version__ = "0.1.0"
