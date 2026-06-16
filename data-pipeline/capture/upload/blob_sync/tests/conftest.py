"""Pytest setup for blob_sync: inject the package root and stub the Azure SDK."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def _install_azure_stub() -> None:
    """Register a minimal ``azure.storage.blob`` so imports need no real SDK."""
    if "azure.storage.blob" in sys.modules:
        return
    azure = sys.modules.setdefault("azure", types.ModuleType("azure"))
    storage = sys.modules.setdefault("azure.storage", types.ModuleType("azure.storage"))
    blob = types.ModuleType("azure.storage.blob")

    class _StubContainerClient:
        @classmethod
        def from_container_url(cls, url: str) -> _StubContainerClient:
            return cls()

        def upload_blob(self, *, name: str, data: object, overwrite: bool) -> None:
            return None

        def get_container_properties(self) -> dict[str, str]:
            return {}

    blob.ContainerClient = _StubContainerClient
    azure.storage = storage
    storage.blob = blob
    sys.modules["azure.storage.blob"] = blob


_install_azure_stub()
