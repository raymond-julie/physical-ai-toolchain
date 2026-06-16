"""Detect when a recorded session is fully encoded and safe to upload.

The dual recorder writes each episode as a LeRobot dataset under
``session_<timestamp>/``: the parquet data first, then the camera ``.mp4``
files under ``videos/``, using temporary ``tmp*`` directories while encoding.

A session is ready to upload when all of the following hold:

1. ``meta/info.json`` exists.
2. ``data/`` contains at least one ``.parquet`` file.
3. ``videos/`` contains at least one ``.mp4`` (when ``require_videos``).
4. No file under the session changed within ``settle_seconds`` (quiescence).
"""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Iterable
from pathlib import Path

UPLOADED_MARKER = ".uploaded"
SESSION_PREFIX = "session_"


def is_excluded(name: str, exclude_globs: Iterable[str]) -> bool:
    """Return True when a file or directory name matches any exclusion glob."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in exclude_globs)


def iter_session_dirs(source_dir: Path) -> list[Path]:
    """Return ``session_*`` subdirectories of ``source_dir`` sorted by name."""
    if not source_dir.is_dir():
        return []
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir() and path.name.startswith(SESSION_PREFIX)
    )


def newest_mtime(root: Path, exclude_globs: Iterable[str]) -> float:
    """Return the most recent mtime of any non-excluded file under ``root``.

    Returns ``0.0`` when the tree contains no eligible files.
    """
    exclude_globs = list(exclude_globs)
    newest = 0.0
    for path in root.rglob("*"):
        if any(is_excluded(part, exclude_globs) for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        newest = max(newest, mtime)
    return newest


def is_already_uploaded(session_dir: Path) -> bool:
    """Return True when the session carries an ``.uploaded`` marker file."""
    return (session_dir / UPLOADED_MARKER).is_file()


def has_expected_outputs(session_dir: Path, require_videos: bool) -> bool:
    """Return True when the structural markers of a completed save are present."""
    if not (session_dir / "meta" / "info.json").is_file():
        return False
    if not any((session_dir / "data").rglob("*.parquet")):
        return False
    if require_videos:
        videos = session_dir / "videos"
        if not videos.is_dir() or not any(videos.rglob("*.mp4")):
            return False
    return True


def is_session_ready(
    session_dir: Path,
    *,
    settle_seconds: float,
    require_videos: bool,
    exclude_globs: Iterable[str],
    now: float | None = None,
) -> bool:
    """Return True when a session is fully encoded, quiescent, and not uploaded."""
    if is_already_uploaded(session_dir):
        return False
    if not has_expected_outputs(session_dir, require_videos):
        return False
    now = time.time() if now is None else now
    newest = newest_mtime(session_dir, exclude_globs)
    if newest == 0.0:
        return False
    return (now - newest) >= settle_seconds
