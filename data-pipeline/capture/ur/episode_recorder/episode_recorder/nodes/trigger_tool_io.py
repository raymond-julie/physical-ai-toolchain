#!/usr/bin/env python3
"""Physical-button (Digital Input) trigger.

Subscribes to a configurable Bool input topic (e.g.
``/robot1/digital_input/di0``) — typically a robot tool digital
input published by :mod:`episode_recorder.nodes.robot_reader`.

On every *rising* edge of that input (debounced), this node publishes
``!current`` to ``/recorder/active``, where ``current`` is the latest
value seen on ``/recorder/active``. The node therefore stays in sync
with whatever other trigger (e.g. the GUI) publishes, so multiple
trigger sources can coexist without state divergence.

If no physical button is wired, simply do not start this node — the
GUI trigger (:mod:`episode_recorder.nodes.trigger_gui`) continues to
work standalone.
"""

from __future__ import annotations

import contextlib
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


class ToolIoTriggerNode(Node):
    """Debounced rising-edge -> toggle of ``/recorder/active``."""

    def __init__(self) -> None:
        super().__init__("trigger_tool_io")

        self.declare_parameter("input_topic", "/robot1/digital_input/di0")
        self.declare_parameter("debounce_seconds", 0.5)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.debounce = float(self.get_parameter("debounce_seconds").value)

        active_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        input_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(Bool, "/recorder/active", active_qos)
        self.create_subscription(Bool, "/recorder/active", self._on_active, active_qos)
        self.create_subscription(Bool, self.input_topic, self._on_input, input_qos)

        self._current_active = False
        self._prev_input = False
        self._last_fire = 0.0

        self.get_logger().info(
            f"Tool I/O trigger ready — listening on {self.input_topic} "
            f"(debounce={self.debounce}s)"
        )

    def _on_active(self, msg: Bool) -> None:
        self._current_active = bool(msg.data)

    def _on_input(self, msg: Bool) -> None:
        val = bool(msg.data)
        rising = val and not self._prev_input
        self._prev_input = val
        if not rising:
            return
        now = time.monotonic()
        if (now - self._last_fire) < self.debounce:
            return
        self._last_fire = now
        new_state = not self._current_active
        out = Bool()
        out.data = new_state
        self.pub.publish(out)
        self.get_logger().info(
            f'{self.input_topic} rising edge -> /recorder/active = '
            f'{"ON" if new_state else "OFF"}'
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ToolIoTriggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
