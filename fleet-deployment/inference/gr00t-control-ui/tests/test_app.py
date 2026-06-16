"""Behavior tests for the GR00T control UI's pure logic.

Two security/behavior-critical pieces are covered without a cluster or server:

* the camera-id SSRF guard -- the ``^[A-Za-z0-9_.-]+$`` regex that constrains the
  proxied camera path so it cannot be steered to an arbitrary upstream URL, and
* the deployment-scale request shaping -- Start/Stop scaling the target Deployment
  to 1/0 and the desired/ready replica counts mapping to a stopped/starting/running
  state.

The cluster (``kubernetes``), HTTP (``httpx``) and web-framework (``fastapi``)
dependencies are stubbed in ``conftest.py`` before this module imports the app, so
each endpoint is a plain callable invoked directly against a mocked AppsV1Api.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# gr00t-control-ui is hyphenated, so the app cannot be imported as a normal
# package. Load it by file path; conftest has already installed the stubs.
_SUT_PATH = Path(__file__).resolve().parents[1] / "app.py"
_spec = importlib.util.spec_from_file_location("gr00t_control_ui_app", _SUT_PATH)
sut = importlib.util.module_from_spec(_spec)
sys.modules["gr00t_control_ui_app"] = sut
_spec.loader.exec_module(sut)


def _make_deployment(replicas: int, ready: int, available: int | None = None) -> MagicMock:
    dep = MagicMock()
    dep.spec.replicas = replicas
    dep.status.ready_replicas = ready
    dep.status.available_replicas = ready if available is None else available
    return dep


# ---------------------------------------------------------------------------
# Camera-id SSRF guard: ^[A-Za-z0-9_.-]+$
# ---------------------------------------------------------------------------
class TestCameraIdValidation:
    @pytest.mark.parametrize(
        "cam_id",
        ["CV3H4600001E", "cam_high", "color-0", "a.b_c-1", "0", "Camera.01"],
    )
    def test_accepts_valid_serials(self, cam_id):
        assert sut._CAM_ID_RE.match(cam_id)

    @pytest.mark.parametrize(
        "cam_id",
        [
            "",
            "../etc/passwd",
            "cam/../secret",
            "a/b",
            "http://169.254.169.254/latest",
            "id with space",
            "id?x=1",
            "a;b",
            "café",
            "host:8000",
        ],
    )
    def test_rejects_ssrf_and_traversal(self, cam_id):
        assert sut._CAM_ID_RE.match(cam_id) is None

    @pytest.mark.parametrize("cam_id", ["../etc/passwd", "a/b", "http://evil/x", "id with space"])
    def test_snapshot_endpoint_rejects_invalid_id(self, cam_id):
        resp = asyncio.run(sut.camera_snapshot(cam_id))
        assert resp.status_code == 400

    @pytest.mark.parametrize("cam_id", ["../etc/passwd", "a/b", "http://evil/x"])
    def test_stream_endpoint_rejects_invalid_id(self, cam_id):
        resp = asyncio.run(sut.camera_stream(cam_id))
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Deployment scale (Start/Stop -> replicas 1/0) and status shaping
# ---------------------------------------------------------------------------
class TestDeploymentScale:
    def setup_method(self) -> None:
        sut.apps.reset_mock()

    def test_start_scales_to_one(self):
        sut.apps.read_namespaced_deployment.return_value = _make_deployment(replicas=1, ready=1)
        result = sut.api_start()

        sut.apps.patch_namespaced_deployment_scale.assert_called_once_with(
            sut.DEPLOYMENT,
            sut.NAMESPACE,
            {"spec": {"replicas": 1}},
        )
        assert result.content["desiredReplicas"] == 1
        assert result.content["state"] == "running"

    def test_stop_scales_to_zero(self):
        sut.apps.read_namespaced_deployment.return_value = _make_deployment(replicas=0, ready=0)
        result = sut.api_stop()

        sut.apps.patch_namespaced_deployment_scale.assert_called_once_with(
            sut.DEPLOYMENT,
            sut.NAMESPACE,
            {"spec": {"replicas": 0}},
        )
        assert result.content["desiredReplicas"] == 0
        assert result.content["state"] == "stopped"

    def test_status_stopped_when_zero_replicas(self):
        sut.apps.read_namespaced_deployment.return_value = _make_deployment(replicas=0, ready=0)
        assert sut._status()["state"] == "stopped"

    def test_status_starting_when_desired_but_not_ready(self):
        sut.apps.read_namespaced_deployment.return_value = _make_deployment(replicas=1, ready=0)
        status = sut._status()
        assert status["state"] == "starting"
        assert status["desiredReplicas"] == 1
        assert status["readyReplicas"] == 0

    def test_status_running_when_ready(self):
        sut.apps.read_namespaced_deployment.return_value = _make_deployment(replicas=1, ready=1, available=1)
        status = sut._status()
        assert status["state"] == "running"
        assert status["availableReplicas"] == 1

    def test_status_reports_target_identifiers(self):
        sut.apps.read_namespaced_deployment.return_value = _make_deployment(replicas=1, ready=1)
        status = sut._status()
        assert status["deployment"] == sut.DEPLOYMENT
        assert status["namespace"] == sut.NAMESPACE
