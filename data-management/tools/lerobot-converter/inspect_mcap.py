#!/usr/bin/env python
"""Inspect an MCAP file: list channels, schemas, encodings, and message counts.

Prints the channel/schema table, recording statistics (message count, duration),
and per-topic message counts -- the first thing to run when a recording looks
wrong before conversion.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def inspect(path: Path) -> None:
    """Print channel, schema, statistics, and per-topic counts for an MCAP file."""
    from mcap.reader import make_reader

    with path.open("rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        print("=== Channels / Schemas ===")
        for channel in summary.channels.values():
            schema = summary.schemas.get(channel.schema_id)
            name = schema.name if schema else "(none)"
            encoding = schema.encoding if schema else ""
            print(
                f"  topic={channel.topic!r:40} "
                f"msg_encoding={channel.message_encoding!r:12} schema={name!r} ({encoding})"
            )
        stats = summary.statistics
        if stats:
            duration_s = (stats.message_end_time - stats.message_start_time) / 1e9
            print("\n=== Statistics ===")
            print(f"  total messages: {stats.message_count}")
            print(f"  start: {stats.message_start_time}  end: {stats.message_end_time}")
            print(f"  duration: {duration_s:.2f}s")

    with path.open("rb") as f:
        reader = make_reader(f)
        counts = defaultdict(int)
        for _schema, channel, _message in reader.iter_messages():
            counts[channel.topic] += 1
    print("\n=== Per-topic message counts ===")
    for topic, count in sorted(counts.items()):
        print(f"  {topic:45} {count}")


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
