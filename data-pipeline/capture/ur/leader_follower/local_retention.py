#!/usr/bin/env python3
"""Local-disk retention loop for LeRobot recordings synced via ACSA.

ACSA mirrors the on-disk dataset under ``${LEROBOT_ROOT}/${LEROBOT_REPO_ID}``
to an Azure Blob container asynchronously, but it does NOT delete the local
copies. This timer node bounds local disk usage on the edge node by deleting
per-episode parquet and mp4 chunks older than ``LEROBOT_RETENTION_DAYS``.
Metadata files (``meta/info.json``, ``meta/stats.json``, ``meta/tasks.jsonl``)
are preserved forever so downstream tooling can still describe historical
episodes by querying blob storage.

Env vars (all optional):

* ``LEROBOT_ROOT``                       Dataset root (default: ``./recordings_lerobot``).
* ``LEROBOT_REPO_ID``                    Repo id under the root (default: ``local/ur5_mirror``).
* ``LEROBOT_RETENTION_DAYS``             Age threshold in days (default: 7).
* ``LEROBOT_CLEANUP_INTERVAL_MINUTES``   Timer interval (default: 60).
* ``LEROBOT_DRY_RUN``                    ``1`` = log deletions without removing files.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

import rclpy
from rclpy.node import Node

DEFAULT_INTERVAL_MIN = 60
DEFAULT_RETENTION_DAYS = 7
_SECONDS_PER_DAY = 86400
_RETENTION_PATTERNS: tuple[str, ...] = (
    "data/chunk-*/episode_*.parquet",
    "videos/chunk-*/observation.images.*/episode_*.mp4",
)


def compute_cutoff(now: float, days: float) -> float:
    """Return the mtime threshold below which a file is considered expired."""
    return now - days * _SECONDS_PER_DAY


def select_expired(base: Path, patterns: Sequence[str], cutoff: float) -> list[Path]:
    """Return per-episode chunk files under ``base`` whose mtime predates ``cutoff``."""
    expired: list[Path] = []
    for pattern in patterns:
        for path in base.glob(pattern):
            try:
                if path.stat().st_mtime < cutoff:
                    expired.append(path)
            except FileNotFoundError:
                continue
    return expired


class LocalRetention(Node):
    """Timer node that prunes aged dataset chunks while keeping metadata."""

    def __init__(self) -> None:
        super().__init__("local_retention")
        self.root = Path(os.environ.get("LEROBOT_ROOT", "./recordings_lerobot"))
        self.repo_id = os.environ.get("LEROBOT_REPO_ID", "local/ur5_mirror")
        self.days = int(os.environ.get("LEROBOT_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))
        self.dry = bool(int(os.environ.get("LEROBOT_DRY_RUN", "0")))
        interval_min = int(os.environ.get("LEROBOT_CLEANUP_INTERVAL_MINUTES", DEFAULT_INTERVAL_MIN))
        self.get_logger().info(
            f"local_retention: root={self.root} repo_id={self.repo_id} "
            f"days={self.days} interval_min={interval_min} dry={self.dry}"
        )
        self.create_timer(interval_min * 60.0, self._tick)
        # Run once immediately so the first cleanup happens at startup.
        self._tick()

    def _tick(self) -> None:
        base = self.root / self.repo_id
        if not base.exists():
            self.get_logger().debug(f"local_retention: {base} does not exist yet")
            return
        cutoff = compute_cutoff(time.time(), self.days)
        deleted = 0
        for path in select_expired(base, _RETENTION_PATTERNS, cutoff):
            self.get_logger().info(f"{'DRY ' if self.dry else ''}retention: removing {path}")
            if self.dry:
                deleted += 1
                continue
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                self.get_logger().warn(f"local_retention: skip {path}: {exc}")
        self.get_logger().info(f"local_retention: deleted={deleted} cutoff_days={self.days}")


def main() -> None:
    rclpy.init()
    node = LocalRetention()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
