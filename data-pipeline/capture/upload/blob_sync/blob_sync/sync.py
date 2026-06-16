"""Sync orchestrator: find ready sessions, upload them, mark them done."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import Config
from .stability import UPLOADED_MARKER, is_session_ready, iter_session_dirs
from .uploader import BlobUploader

_LOGGER = logging.getLogger(__name__)


def _write_marker(session_dir: Path, file_count: int, container_redacted: str) -> None:
    marker = {
        "uploaded_at": datetime.now(UTC).isoformat(),
        "files": file_count,
        "container": container_redacted,
    }
    (session_dir / UPLOADED_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")


def sync_once(cfg: Config, uploader: BlobUploader) -> int:
    """Upload every ready session exactly once and return the upload count."""
    sessions = iter_session_dirs(cfg.source_dir)
    if not sessions:
        _LOGGER.debug("No session_* directories under %s", cfg.source_dir)
        return 0

    uploaded = 0
    for session_dir in sessions:
        if not is_session_ready(
            session_dir,
            settle_seconds=cfg.settle_seconds,
            require_videos=cfg.require_videos,
            exclude_globs=cfg.exclude_globs,
        ):
            continue
        _LOGGER.info("Session ready: %s", session_dir.name)
        try:
            count = uploader.upload_session(session_dir, cfg.exclude_globs)
        except Exception:
            _LOGGER.exception("Upload failed for %s; will retry next pass.", session_dir.name)
            continue
        _write_marker(session_dir, count, cfg.container_url_redacted)
        uploaded += 1
    return uploaded


def watch(cfg: Config, uploader: BlobUploader) -> None:
    """Continuously poll for ready sessions until interrupted."""
    _LOGGER.info(
        "Watching %s every %.0fs (settle %.0fs) -> %s",
        cfg.source_dir,
        cfg.poll_interval_seconds,
        cfg.settle_seconds,
        cfg.container_url_redacted,
    )
    while True:
        try:
            sync_once(cfg, uploader)
        except Exception:
            _LOGGER.exception("Sync pass failed; continuing.")
        time.sleep(cfg.poll_interval_seconds)
