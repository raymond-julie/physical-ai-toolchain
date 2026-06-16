#!/usr/bin/env python
"""Report the per-joint motion range of the left vs right follower arms.

Reads ``joint_states`` for each follower arm from an MCAP recording and prints
the per-dimension travel (max - min), a quick check that an episode actually
contains motion before conversion.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

_LOGGER = logging.getLogger(__name__)

FOLLOWER_TOPICS = [
    "/arm_left_follower/arm/joint_states",
    "/arm_right_follower/arm/joint_states",
]


def joint_motion_range(positions: Sequence[Sequence[float]]) -> np.ndarray:
    """Per-dimension ``max - min`` across all sampled joint position vectors."""
    arr = np.asarray(positions, dtype=np.float64)
    return arr.max(axis=0) - arr.min(axis=0)


def read_joint_positions(path: Path, topics: list[str]) -> dict[str, list[list[float]]]:
    """Collect joint position vectors per topic from an MCAP file."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory as Ros2Decoder

    ros2 = Ros2Decoder()
    data = {topic: [] for topic in topics}
    with path.open("rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=topics):
            decoded = ros2.decoder_for("cdr", schema)(message.data)
            data[channel.topic].append(list(decoded.position))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Path to the MCAP file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.path.is_file():
        _LOGGER.error("MCAP file not found: %s", args.path)
        return 1

    data = read_joint_positions(args.path, FOLLOWER_TOPICS)
    for topic, positions in data.items():
        if not positions:
            print(f"{topic} n=0 (no messages)")
            continue
        ranges = joint_motion_range(positions)
        print(f"{topic} n={len(positions)}")
        print("  range per dim:", np.round(ranges, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
