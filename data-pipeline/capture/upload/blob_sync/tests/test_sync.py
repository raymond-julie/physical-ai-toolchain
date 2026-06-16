"""Tests for blob_sync upload orchestration and blob-name mapping."""

from __future__ import annotations

import json
from pathlib import Path

from blob_sync.config import Config
from blob_sync.sync import sync_once
from blob_sync.uploader import BlobUploader


class _RecordingUploader:
    def __init__(self) -> None:
        self.uploaded: list[str] = []

    def upload_session(self, session_dir: Path, exclude_globs: object) -> int:
        self.uploaded.append(session_dir.name)
        return 3


def _ready_session(root: Path, name: str) -> Path:
    session = root / name
    (session / "meta").mkdir(parents=True)
    (session / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (session / "data").mkdir()
    (session / "data" / "e.parquet").write_bytes(b"x")
    videos = session / "videos" / "cam"
    videos.mkdir(parents=True)
    (videos / "f.mp4").write_bytes(b"x")
    return session


def _config(source_dir: Path) -> Config:
    return Config(
        source_dir=source_dir,
        container_url="https://a.blob.core.windows.net/c?sig=x",
        blob_prefix="ur_dual_recorder",
        settle_seconds=0.0,
    )


class TestBlobName:
    def test_includes_prefix_session_and_relative(self) -> None:
        uploader = BlobUploader("https://a.blob.core.windows.net/c?sig=x", "ur")
        name = uploader._blob_name("session_1", Path("videos/cam/f.mp4"))
        assert name == "ur/session_1/videos/cam/f.mp4"

    def test_empty_prefix(self) -> None:
        uploader = BlobUploader("https://a.blob.core.windows.net/c?sig=x", "")
        assert uploader._blob_name("s", Path("meta/info.json")) == "s/meta/info.json"


class TestIterUploadFiles:
    def test_excludes_marker_and_tmp(self, tmp_path: Path) -> None:
        session = _ready_session(tmp_path, "session_1")
        (session / ".uploaded").write_text("{}", encoding="utf-8")
        (session / "tmpdir").mkdir()
        (session / "tmpdir" / "x.bin").write_bytes(b"x")
        uploader = BlobUploader("https://a.blob.core.windows.net/c?sig=x", "")
        rels = sorted(
            path.relative_to(session).as_posix()
            for path in uploader._iter_upload_files(session, [".uploaded", "tmp*"])
        )
        assert ".uploaded" not in rels
        assert all(not rel.startswith("tmpdir/") for rel in rels)
        assert "meta/info.json" in rels


class TestSyncOnce:
    def test_uploads_ready_sessions_and_writes_marker(self, tmp_path: Path) -> None:
        _ready_session(tmp_path, "session_1")
        uploader = _RecordingUploader()
        count = sync_once(_config(tmp_path), uploader)
        assert count == 1
        assert uploader.uploaded == ["session_1"]
        marker = json.loads((tmp_path / "session_1" / ".uploaded").read_text(encoding="utf-8"))
        assert marker["files"] == 3
        assert "sas-redacted" in marker["container"]

    def test_skips_unready_session(self, tmp_path: Path) -> None:
        session = _ready_session(tmp_path, "session_2")
        (session / "data" / "e.parquet").unlink()
        uploader = _RecordingUploader()
        assert sync_once(_config(tmp_path), uploader) == 0
        assert uploader.uploaded == []
