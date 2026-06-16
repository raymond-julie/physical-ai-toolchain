"""Stub the control UI's cluster / HTTP / web-framework dependencies.

The logic under test -- the camera-id SSRF regex and the deployment-scale request
shaping -- needs neither a real Kubernetes cluster nor a running web server, so
``kubernetes``, ``httpx`` and ``fastapi`` are replaced with lightweight stand-ins
in ``sys.modules`` before the module under test is imported. The FastAPI stub's
route decorators return the handler unchanged, so each endpoint stays a plain
callable the tests can invoke directly; the response stubs capture
``status_code`` / ``content`` / ``media_type`` for assertions.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


class _ApiException(Exception):
    """Stand-in for ``kubernetes.client.rest.ApiException``."""

    def __init__(self, reason: str = "", status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


class _ConfigException(Exception):
    """Stand-in for ``kubernetes.config.ConfigException``."""


class _StubResponse:
    """Capture the response args the handlers pass so tests can assert on them."""

    def __init__(
        self,
        content: object = None,
        status_code: int = 200,
        media_type: str | None = None,
        **_kwargs: object,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.media_type = media_type


class _StubFastAPI:
    """Minimal FastAPI stand-in whose route decorators are identity functions."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.routes: list[object] = []

    def _decorator(self, *_args: object, **_kwargs: object):
        def wrap(func):
            self.routes.append(func)
            return func

        return wrap

    get = _decorator
    post = _decorator


def _install_stubs() -> None:
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _StubFastAPI

    fastapi_responses = types.ModuleType("fastapi.responses")
    for name in ("HTMLResponse", "JSONResponse", "Response", "StreamingResponse"):
        setattr(fastapi_responses, name, type(name, (_StubResponse,), {}))
    fastapi.responses = fastapi_responses

    httpx = types.ModuleType("httpx")
    httpx.AsyncClient = MagicMock()
    httpx.Timeout = MagicMock()

    k8s_rest = types.ModuleType("kubernetes.client.rest")
    k8s_rest.ApiException = _ApiException

    k8s_client = types.ModuleType("kubernetes.client")
    k8s_client.AppsV1Api = MagicMock()
    k8s_client.CoreV1Api = MagicMock()
    k8s_client.CustomObjectsApi = MagicMock()
    k8s_client.rest = k8s_rest

    k8s_config = types.ModuleType("kubernetes.config")
    k8s_config.load_incluster_config = MagicMock()
    k8s_config.load_kube_config = MagicMock()
    k8s_config.ConfigException = _ConfigException

    k8s = types.ModuleType("kubernetes")
    k8s.client = k8s_client
    k8s.config = k8s_config

    stubs = {
        "fastapi": fastapi,
        "fastapi.responses": fastapi_responses,
        "httpx": httpx,
        "kubernetes": k8s,
        "kubernetes.client": k8s_client,
        "kubernetes.client.rest": k8s_rest,
        "kubernetes.config": k8s_config,
    }
    for name, module in stubs.items():
        sys.modules[name] = module


_install_stubs()
