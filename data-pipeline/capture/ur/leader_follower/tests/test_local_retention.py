"""Tests for the local-retention pure helpers (no ROS, no hardware).

Covers ``compute_cutoff`` (age threshold math) and ``select_expired`` (glob +
mtime selection of per-episode chunk files, leaving newer files untouched).
"""

from __future__ import annotations

import os
from pathlib import Path

from local_retention import _RETENTION_PATTERNS, compute_cutoff, select_expired

_SECONDS_PER_DAY = 86400


class TestComputeCutoff:
    def test_cutoff_subtracts_whole_days(self) -> None:
        assert compute_cutoff(1_000_000.0, 1) == 1_000_000.0 - _SECONDS_PER_DAY

    def test_zero_days_keeps_now(self) -> None:
        assert compute_cutoff(1_000_000.0, 0) == 1_000_000.0

    def test_fractional_days_supported(self) -> None:
        assert compute_cutoff(0.0, 0.5) == -0.5 * _SECONDS_PER_DAY


def _touch(path: Path, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))
    return path


class TestSelectExpired:
    def test_only_files_older_than_cutoff_are_selected(self, tmp_path: Path) -> None:
        cutoff = 1_000_000.0
        old_parquet = _touch(
            tmp_path / "data/chunk-000/episode_000001.parquet", cutoff - 100
        )
        new_parquet = _touch(
            tmp_path / "data/chunk-000/episode_000002.parquet", cutoff + 100
        )
        old_video = _touch(
            tmp_path / "videos/chunk-000/observation.images.cam/episode_000001.mp4",
            cutoff - 100,
        )

        expired = select_expired(tmp_path, _RETENTION_PATTERNS, cutoff)

        assert old_parquet in expired
        assert old_video in expired
        assert new_parquet not in expired

    def test_metadata_files_are_never_matched(self, tmp_path: Path) -> None:
        # meta/ files do not match the per-episode chunk patterns, so they are
        # preserved regardless of age.
        _touch(tmp_path / "meta/info.json", 0.0)
        _touch(tmp_path / "data/chunk-000/episode_000001.parquet", 0.0)

        expired = select_expired(tmp_path, _RETENTION_PATTERNS, cutoff=1_000_000.0)

        assert all(path.suffix == ".parquet" for path in expired)
        assert len(expired) == 1

    def test_no_matches_returns_empty_list(self, tmp_path: Path) -> None:
        assert select_expired(tmp_path, _RETENTION_PATTERNS, cutoff=1.0) == []
