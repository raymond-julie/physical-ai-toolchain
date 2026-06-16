"""Azure Blob Storage uploader for recorded sessions.

Uses a container-level SAS URL so no account key is needed. Each session's
files are uploaded under ``<blob_prefix>/<session_name>/<relative_path>``.
Uploads overwrite existing blobs, so an interrupted run can be retried.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from azure.storage.blob import ContainerClient

from .stability import is_excluded

_LOGGER = logging.getLogger(__name__)


class BlobUploader:
    """Thin wrapper around an Azure ``ContainerClient`` built from a SAS URL."""

    def __init__(self, container_url: str, blob_prefix: str = "") -> None:
        self._client = ContainerClient.from_container_url(container_url)
        self._prefix = blob_prefix.strip("/")

    def _blob_name(self, session_name: str, relative: Path) -> str:
        parts = [part for part in (self._prefix, session_name, relative.as_posix()) if part]
        return "/".join(parts)

    def _iter_upload_files(self, session_dir: Path, exclude_globs: Iterable[str]) -> Iterator[Path]:
        exclude_globs = list(exclude_globs)
        for path in sorted(session_dir.rglob("*")):
            rel = path.relative_to(session_dir)
            if any(is_excluded(part, exclude_globs) for part in rel.parts):
                continue
            if path.is_file():
                yield path

    def upload_session(self, session_dir: Path, exclude_globs: Iterable[str]) -> int:
        """Upload every eligible file in a session and return the file count."""
        session_name = session_dir.name
        count = 0
        for path in self._iter_upload_files(session_dir, exclude_globs):
            rel = path.relative_to(session_dir)
            blob_name = self._blob_name(session_name, rel)
            _LOGGER.debug("Uploading %s -> %s", path, blob_name)
            with path.open("rb") as data:
                self._client.upload_blob(name=blob_name, data=data, overwrite=True)
            count += 1
        _LOGGER.info("Uploaded %d files for %s", count, session_name)
        return count

    def check_access(self) -> None:
        """Validate the SAS URL and container by reading container properties."""
        self._client.get_container_properties()
