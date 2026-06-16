"""Tests for blob_sync configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from blob_sync.config import BlobSyncConfigError, load_config


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLOB_SYNC_CONTAINER_URL", raising=False)


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


class TestLoadConfig:
    def test_minimal_valid(self, tmp_path: Path) -> None:
        cfg_path = _write_config(
            tmp_path / "config.yaml",
            'source_dir: "sessions"\ncontainer_url: "https://acct.blob.core.windows.net/c?sig=x"\n',
        )
        cfg = load_config(cfg_path)
        assert cfg.source_dir == (tmp_path / "sessions").resolve()
        assert cfg.blob_prefix == ""
        assert cfg.settle_seconds == 60.0
        assert cfg.require_videos is True

    def test_defaults_overridden(self, tmp_path: Path) -> None:
        cfg_path = _write_config(
            tmp_path / "config.yaml",
            "\n".join(
                [
                    'source_dir: "s"',
                    'container_url: "https://a.blob.core.windows.net/c?sig=x"',
                    'blob_prefix: "/ur/"',
                    "settle_seconds: 5",
                    "poll_interval_seconds: 2",
                    "require_videos: false",
                ]
            ),
        )
        cfg = load_config(cfg_path)
        assert cfg.blob_prefix == "ur"
        assert cfg.settle_seconds == 5.0
        assert cfg.poll_interval_seconds == 2.0
        assert cfg.require_videos is False

    def test_container_url_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLOB_SYNC_CONTAINER_URL", "https://a.blob.core.windows.net/c?sig=y")
        cfg_path = _write_config(tmp_path / "config.yaml", 'source_dir: "s"\n')
        cfg = load_config(cfg_path)
        assert "sig=y" in cfg.container_url

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(BlobSyncConfigError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_missing_container_url(self, tmp_path: Path) -> None:
        cfg_path = _write_config(tmp_path / "config.yaml", 'source_dir: "s"\n')
        with pytest.raises(BlobSyncConfigError, match="container_url is required"):
            load_config(cfg_path)

    def test_missing_source_dir(self, tmp_path: Path) -> None:
        cfg_path = _write_config(
            tmp_path / "config.yaml",
            'container_url: "https://a.blob.core.windows.net/c?sig=x"\n',
        )
        with pytest.raises(BlobSyncConfigError, match="source_dir is required"):
            load_config(cfg_path)

    def test_non_mapping(self, tmp_path: Path) -> None:
        cfg_path = _write_config(tmp_path / "config.yaml", "- just\n- a\n- list\n")
        with pytest.raises(BlobSyncConfigError, match="YAML mapping"):
            load_config(cfg_path)

    def test_redacted_url(self, tmp_path: Path) -> None:
        cfg_path = _write_config(
            tmp_path / "config.yaml",
            'source_dir: "s"\ncontainer_url: "https://a.blob.core.windows.net/c?sig=secret"\n',
        )
        cfg = load_config(cfg_path)
        assert "secret" not in cfg.container_url_redacted
        assert cfg.container_url_redacted.endswith("<sas-redacted>")
