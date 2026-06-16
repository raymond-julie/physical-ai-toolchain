#!/usr/bin/env python
"""Inspect TCP wrench (WrenchStamped) and gripper-related MCAP topics.

Prints the first wrench/tool-grabbed message on each topic, then summarizes the
left follower gripper joint (index 6) range and tool-grabbed transitions across
the whole recording.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger(__name__)

SAMPLE_TOPICS = [
    "/arm_left_follower/arm/tcp_wrench",
    "/arm_right_follower/arm/tcp_wrench",
    "/arm_left_leader/arm/tcp_wrench",
    "/arm_left_follower/action/SetToolGrabbed",
    "/arm_left_leader/arm/tool_grabbed",
]
GRIPPER_JOINT_INDEX = 6


def inspect(path: Path) -> None:
    """Print wrench samples and a gripper/tool-grabbed summary for an MCAP file."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory as Ros2Decoder

    ros2 = Ros2Decoder()
    want = set(SAMPLE_TOPICS)
    seen = set()
    grip_left = []
    toolgrab_follower = []
    toolgrab_leader = []

    with path.open("rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic == "/arm_left_follower/arm/joint_states":
                decoded = ros2.decoder_for("cdr", schema)(message.data)
                grip_left.append(decoded.position[GRIPPER_JOINT_INDEX])
            elif topic == "/arm_left_follower/action/SetToolGrabbed":
                decoded = ros2.decoder_for("cdr", schema)(message.data)
                toolgrab_follower.append(int(decoded.data))
            elif topic == "/arm_left_leader/arm/tool_grabbed":
                decoded = ros2.decoder_for("cdr", schema)(message.data)
                toolgrab_leader.append(int(decoded.data))

            if topic in want and topic not in seen:
                seen.add(topic)
                print(f"\n===== {topic} (enc={channel.message_encoding}, schema={schema.name}) =====")
                if channel.message_encoding == "json":
                    print(json.dumps(json.loads(message.data), indent=2)[:1500])
                    continue
                decoded = ros2.decoder_for("cdr", schema)(message.data)
                if "tcp_wrench" in topic:
                    force, torque = decoded.wrench.force, decoded.wrench.torque
                    print("force:", force.x, force.y, force.z)
                    print("torque:", torque.x, torque.y, torque.z)
                    print("frame_id:", decoded.header.frame_id)
                else:
                    print("data:", decoded.data)

    print("\n=== gripper (left follower joint[6]) ===")
    grip = np.asarray(grip_left)
    if grip.size:
        print("n:", grip.size, "min:", grip.min(), "max:", grip.max(), "unique sample:", np.unique(np.round(grip, 4))[:10])
    else:
        print("n: 0 (no joint_states)")

    print("\n=== SetToolGrabbed (left follower) ===")
    follower = np.asarray(toolgrab_follower)
    if follower.size:
        print("n:", follower.size, "values:", np.unique(follower), "first->last:", follower[0], follower[-1])
    else:
        print("n: 0")

    print("=== tool_grabbed (left leader) ===")
    leader = np.asarray(toolgrab_leader)
    print("n:", leader.size, "values:", np.unique(leader) if leader.size else [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Path to the MCAP file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.path.is_file():
        _LOGGER.error("MCAP file not found: %s", args.path)
        return 1

    inspect(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
