#!/usr/bin/env python
"""Decode sample MCAP messages; JSON handled manually, CDR via the ROS 2 decoder.

Prints the first message seen on each requested topic, including compressed-image
headers, so the wire encoding of a recording can be verified before conversion.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

DEFAULT_TOPICS = [
    "/arm_left_follower/action/MoveToPosition",
    "/arm_left_follower/action/SetToolGrabbed",
    "/cam_high/camera/left_image/distorted/compressed",
    "/cam_high/camera/left_image/distorted/info",
]


def inspect(path: Path, topics: list[str]) -> None:
    """Print the first decoded message for each requested topic."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory as Ros2Decoder

    want = set(topics)
    seen = set()
    ros2 = Ros2Decoder()
    with path.open("rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic in want and topic not in seen:
                seen.add(topic)
                print(f"\n===== {topic} (enc={channel.message_encoding}) =====")
                if channel.message_encoding == "json":
                    print(json.dumps(json.loads(message.data), indent=2)[:2000])
                else:
                    decoded = ros2.decoder_for("cdr", schema)(message.data)
                    if "compressed" in topic:
                        print(
                            "format:", decoded.format,
                            "| data bytes:", len(decoded.data),
                            "| first bytes:", bytes(decoded.data[:4]),
                        )
                        print(
                            "stamp:", decoded.header.stamp.sec, decoded.header.stamp.nanosec,
                            "frame:", decoded.header.frame_id,
                        )
                    elif "info" in topic:
                        print("w x h:", decoded.width, decoded.height)
                    elif "SetToolGrabbed" in topic:
                        print("data:", decoded.data)
            if len(seen) == len(want):
                break
    print("\nNot found:", want - seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Path to the MCAP file")
    parser.add_argument(
        "--topics",
        nargs="+",
        default=DEFAULT_TOPICS,
        help="Topics to sample (default: action, tool-grabbed, and camera topics)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.path.is_file():
        _LOGGER.error("MCAP file not found: %s", args.path)
        return 1

    inspect(args.path, args.topics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
