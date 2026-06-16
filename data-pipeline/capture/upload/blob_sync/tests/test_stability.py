"""Tests for blob_sync session-readiness (quiescence) logic."""

from __future__ import annotations

import os
import time
from pathlib import Path

from blob_sync.stability import (
    has_expected_outputs,
    is_already_uploaded,
    is_excluded,
    is_session_ready,
    iter_session_dirs,
    newest_mtime,
)


def _make_session(root: Path, name: str = "session_001", *, with_video: bool = True) -> Path:
    session = root / name
    (session / "meta").mkdir(parents=True)
    (session / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (session / "data").mkdir()
    (session / "data" / "episode_000.parquet").write_bytes(b"x")
    if with_video:
        video_dir = session / "videos" / "observation.images.cam_high"
        video_dir.mkdir(parents=True)
        (video_dir / "file-000.mp4").write_bytes(b"x")
    return session


class TestIsExcluded:
    def test_matches_glob(self) -> None:
        assert is_excluded("tmp123", ["tmp*", "*.tmp"]) is True

    def test_no_match(self) -> None:
        assert is_excluded("data", ["tmp*", "*.tmp"]) is False


class TestIterSessionDirs:
    def test_missing_source_returns_empty(self, tmp_path: Path) -> None:
        assert iter_session_dirs(tmp_path / "nope") == []

    def test_only_session_prefixed_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "session_a").mkdir()
        (tmp_path / "session_b").mkdir()
        (tmp_path / "other").mkdir()
        (tmp_path / "session_file.txt").write_text("x", encoding="utf-8")
        names = [path.name for path in iter_session_dirs(tmp_path)]
        assert names == ["session_a", "session_b"]


class TestNewestMtime:
    def test_empty_tree_returns_zero(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        assert newest_mtime(tmp_path / "empty", []) == 0.0

    def test_excluded_files_ignored(self, tmp_path: Path) -> None:
        marker = tmp_path / ".uploaded"
        marker.write_text("x", encoding="utf-8")
        data = tmp_path / "data.parquet"
        data.write_bytes(b"x")
        os.utime(data, (1000, 1000))
        os.utime(marker, (5000, 5000))
        assert newest_mtime(tmp_path, [".uploaded"]) == 1000.0


class TestHasExpectedOutputs:
    def test_complete_session(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        assert has_expected_outputs(session, require_videos=True) is True

    def test_missing_info_json(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        (session / "meta" / "info.json").unlink()
        assert has_expected_outputs(session, require_videos=True) is False

    def test_missing_parquet(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        (session / "data" / "episode_000.parquet").unlink()
        assert has_expected_outputs(session, require_videos=True) is False

    def test_missing_video_required(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, with_video=False)
        assert has_expected_outputs(session, require_videos=True) is False

    def test_missing_video_not_required(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, with_video=False)
        assert has_expected_outputs(session, require_videos=False) is True


class TestIsSessionReady:
    def test_ready_when_quiescent(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        future = time.time() + 1000
        assert (
            is_session_ready(
                session, settle_seconds=60, require_videos=True, exclude_globs=[], now=future
            )
            is True
        )

    def test_not_ready_when_recent(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        assert (
            is_session_ready(
                session,
                settle_seconds=600,
                require_videos=True,
                exclude_globs=[],
                now=time.time(),
            )
            is False
        )

    def test_not_ready_when_already_uploaded(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        (session / ".uploaded").write_text("{}", encoding="utf-8")
        future = time.time() + 1000
        assert (
            is_session_ready(
                session,
                settle_seconds=60,
                require_videos=True,
                exclude_globs=[".uploaded"],
                now=future,
            )
            is False
        )

    def test_is_already_uploaded_false_without_marker(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        assert is_already_uploaded(session) is False
