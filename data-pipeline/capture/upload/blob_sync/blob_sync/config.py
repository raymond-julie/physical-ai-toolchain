"""Configuration loading for blob_sync.

Settings come from a YAML file (default ``config.yaml`` next to the package).
The container SAS URL is a secret and lives in that git-ignored file or the
``BLOB_SYNC_CONTAINER_URL`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_EXCLUDE_GLOBS = ["tmp*", "*.tmp", ".uploaded"]


class BlobSyncConfigError(RuntimeError):
    """Raised when the blob_sync configuration is missing or invalid."""


@dataclass
class Config:
    """Resolved blob_sync settings."""

    source_dir: Path
    container_url: str
    blob_prefix: str = ""
    settle_seconds: float = 60.0
    poll_interval_seconds: float = 30.0
    require_videos: bool = True
    exclude_globs: list[str] = field(default_factory=lambda: list(_DEFAULT_EXCLUDE_GLOBS))

    @property
    def container_url_redacted(self) -> str:
        """Container URL with the SAS query string masked for logging."""
        base, _, _ = self.container_url.partition("?")
        return f"{base}?<sas-redacted>" if "?" in self.container_url else base


def _resolve_path(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_config(config_path: str | os.PathLike[str]) -> Config:
    """Load and validate a :class:`Config` from a YAML file."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise BlobSyncConfigError(
            f"Config file not found: {path}. Copy config.example.yaml to "
            f"config.yaml and fill in your container SAS URL."
        )

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise BlobSyncConfigError(f"Config file {path} must contain a YAML mapping.")

    container_url = (
        os.environ.get("BLOB_SYNC_CONTAINER_URL") or data.get("container_url") or ""
    ).strip()
    if not container_url:
        raise BlobSyncConfigError(
            "container_url is required (set it in the config file or the "
            "BLOB_SYNC_CONTAINER_URL environment variable)."
        )

    raw_source = data.get("source_dir")
    if not raw_source:
        raise BlobSyncConfigError("source_dir is required in the config file.")

    return Config(
        source_dir=_resolve_path(str(raw_source), path.parent),
        container_url=container_url,
        blob_prefix=str(data.get("blob_prefix", "")).strip().strip("/"),
        settle_seconds=float(data.get("settle_seconds", 60.0)),
        poll_interval_seconds=float(data.get("poll_interval_seconds", 30.0)),
        require_videos=bool(data.get("require_videos", True)),
        exclude_globs=list(data.get("exclude_globs", _DEFAULT_EXCLUDE_GLOBS)),
    )
