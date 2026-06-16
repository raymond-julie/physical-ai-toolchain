"""Control + observability UI for the GR00T inference deployment.

Exposes Start / Stop / Status plus live observability panels so an operator can
evaluate how a run is going from a single page:

* Start / Stop / Status: "Start" scales the target Deployment to 1, "Stop"
  scales it to 0 (freeing the GPU; cached weights stay on the PVC).
* Live camera streams: reverse-proxied from the ur-camera-streamer service so a
  single port-forward to this UI surfaces every Orbbec MJPEG feed.
* Resource graphs: CPU / memory of the inference pod over time, read from the
  metrics-server (metrics.k8s.io).
* GPU allocation: best-effort node GPU capacity vs. in-use (Jetson Orin exposes
  no per-pod utilization to in-cluster pods, so this reports allocation, not %).

Runs in-cluster using the mounted ServiceAccount token. RBAC restricts this
ServiceAccount to the single target Deployment (get/scale), read-only pod and
pod-metrics access in its namespace, and read-only node access for GPU counts.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from kubernetes import client, config
from kubernetes.client.rest import ApiException

NAMESPACE = os.environ.get("TARGET_NAMESPACE", "default")
DEPLOYMENT = os.environ.get("TARGET_DEPLOYMENT", "gr00t-gr00t-inference")
# Base URL of the ur-camera-streamer (hostNetwork on the same node). The chart
# injects http://<hostIP>:8000 via the downward API.
CAMERA_STREAMER_URL = os.environ.get("CAMERA_STREAMER_URL", "http://127.0.0.1:8000").rstrip("/")
# nvidia GPU resource name advertised by the device plugin.
GPU_RESOURCE = os.environ.get("GPU_RESOURCE_NAME", "nvidia.com/gpu")
# Optional path (inside the container) to the Jetson GPU load file. When mounted
# from the host (/sys/devices/platform/gpu.0/load), it reports load on a 0-1000
# scale, letting the GPU panel show real utilization instead of only allocation.
GPU_LOAD_FILE = os.environ.get("GPU_LOAD_FILE", "")

# Camera IDs are device serials; constrain them so proxied paths can't be used
# to reach arbitrary upstream URLs (SSRF / path traversal).
_CAM_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# In-cluster config uses the mounted ServiceAccount; fall back to kubeconfig
# for local development.
try:
    config.load_incluster_config()
except config.ConfigException:
    config.load_kube_config()

apps = client.AppsV1Api()
core = client.CoreV1Api()
custom = client.CustomObjectsApi()

# Shared async client for proxying camera streams/snapshots. No total timeout so
# long-lived MJPEG streams are not cut off; connect timeout still applies.
_http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))

app = FastAPI(title="GR00T Control")


def _status() -> dict:
    dep = apps.read_namespaced_deployment(DEPLOYMENT, NAMESPACE)
    spec_replicas = dep.spec.replicas or 0
    ready = dep.status.ready_replicas or 0
    available = dep.status.available_replicas or 0
    if spec_replicas == 0:
        state = "stopped"
    elif ready >= 1:
        state = "running"
    else:
        state = "starting"
    return {
        "deployment": DEPLOYMENT,
        "namespace": NAMESPACE,
        "desiredReplicas": spec_replicas,
        "readyReplicas": ready,
        "availableReplicas": available,
        "state": state,
    }


def _scale(replicas: int) -> dict:
    apps.patch_namespaced_deployment_scale(
        DEPLOYMENT,
        NAMESPACE,
        {"spec": {"replicas": replicas}},
    )
    return _status()


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(_status())


@app.post("/api/start")
def api_start() -> JSONResponse:
    return JSONResponse(_scale(1))


@app.post("/api/stop")
def api_stop() -> JSONResponse:
    return JSONResponse(_scale(0))


def _parse_cpu(value: str) -> float:
    """Kubernetes CPU quantity -> millicores."""
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000.0
    if value.endswith("u"):
        return float(value[:-1]) / 1_000.0
    if value.endswith("m"):
        return float(value[:-1])
    return float(value) * 1000.0


_MEM_UNITS = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "K": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
}


def _parse_mem(value: str) -> float:
    """Kubernetes memory quantity -> bytes."""
    for suffix, mult in _MEM_UNITS.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * mult
    return float(value)


def _target_selector() -> str:
    """Label selector matching the inference deployment's pods."""
    dep = apps.read_namespaced_deployment(DEPLOYMENT, NAMESPACE)
    labels = (dep.spec.selector.match_labels or {}) if dep.spec.selector else {}
    return ",".join(f"{k}={v}" for k, v in labels.items())


@app.get("/api/metrics")
def api_metrics() -> JSONResponse:
    """CPU/memory usage of the inference pod(s) from the metrics-server."""
    pods: list[dict] = []
    total_cpu = 0.0
    total_mem = 0.0
    error = None
    try:
        selector = _target_selector()
        result = custom.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=NAMESPACE,
            plural="pods",
            label_selector=selector,
        )
        for item in result.get("items", []):
            cpu = 0.0
            mem = 0.0
            for container in item.get("containers", []):
                usage = container.get("usage", {})
                cpu += _parse_cpu(usage.get("cpu", "0"))
                mem += _parse_mem(usage.get("memory", "0"))
            pods.append(
                {
                    "name": item.get("metadata", {}).get("name", "?"),
                    "cpuMillicores": round(cpu),
                    "memBytes": int(mem),
                }
            )
            total_cpu += cpu
            total_mem += mem
    except ApiException as exc:
        error = f"metrics unavailable: {exc.reason}"

    return JSONResponse(
        {
            "pods": pods,
            "totalCpuMillicores": round(total_cpu),
            "totalMemBytes": int(total_mem),
            "ts": time.time(),
            "error": error,
        }
    )


@app.get("/api/gpu")
def api_gpu() -> JSONResponse:
    """Best-effort GPU allocation across nodes (capacity vs. in-use)."""
    capacity = 0
    allocatable = 0
    error = None
    try:
        for node in core.list_node().items:
            cap = (node.status.capacity or {}).get(GPU_RESOURCE)
            alloc = (node.status.allocatable or {}).get(GPU_RESOURCE)
            if cap:
                capacity += int(cap)
            if alloc:
                allocatable += int(alloc)
    except ApiException as exc:
        error = f"node read failed: {exc.reason}"

    state = _status()
    # Each inference replica requests one GPU, so desired replicas == GPUs held.
    in_use = state["desiredReplicas"]

    # Real utilization, when the host GPU load file is mounted (Jetson: 0-1000).
    utilization = None
    if GPU_LOAD_FILE:
        try:
            with open(GPU_LOAD_FILE) as fh:
                raw = fh.read().strip()
            utilization = round(int(raw) / 10.0, 1)
        except (OSError, ValueError):
            utilization = None

    return JSONResponse(
        {
            "resourceName": GPU_RESOURCE,
            "capacity": capacity,
            "allocatable": allocatable,
            "inUse": in_use,
            "utilizationPercent": utilization,
            "state": state["state"],
            "error": error,
        }
    )


@app.get("/api/cameras")
async def api_cameras() -> JSONResponse:
    """Catalog of cameras from ur-camera-streamer, with URLs rewritten to proxy
    back through this UI so a single port-forward serves every stream."""
    try:
        resp = await _http.get(f"{CAMERA_STREAMER_URL}/api/cameras", timeout=5.0)
        data = resp.json()
    except Exception as exc:
        # Surface any upstream failure to the UI rather than 500-ing.
        return JSONResponse({"cameras": [], "error": str(exc)})

    cameras = []
    for cam in data.get("cameras", []):
        cam_id = cam.get("id", "")
        cameras.append(
            {
                **cam,
                "stream_url": f"camera/stream/{cam_id}",
                "snapshot_url": f"camera/snapshot/{cam_id}",
            }
        )
    return JSONResponse({"cameras": cameras})


@app.get("/camera/snapshot/{cam_id}")
async def camera_snapshot(cam_id: str) -> Response:
    if not _CAM_ID_RE.match(cam_id):
        return Response(content="invalid camera id", status_code=400)
    try:
        resp = await _http.get(f"{CAMERA_STREAMER_URL}/snapshot/{cam_id}", timeout=10.0)
    except Exception as exc:
        return Response(content=f"camera unreachable: {exc}", status_code=502)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "image/jpeg"),
    )


@app.get("/camera/stream/{cam_id}")
async def camera_stream(cam_id: str) -> StreamingResponse | Response:
    if not _CAM_ID_RE.match(cam_id):
        return Response(content="invalid camera id", status_code=400)
    url = f"{CAMERA_STREAMER_URL}/stream/{cam_id}"

    async def relay() -> AsyncIterator[bytes]:
        try:
            async with _http.stream("GET", url) as upstream:
                async for chunk in upstream.aiter_raw():
                    yield chunk
        except Exception:
            # Client disconnects end the generator; nothing to surface.
            return

    # The streamer publishes multipart/x-mixed-replace with boundary "frame".
    return StreamingResponse(relay(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_PAGE)


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GR00T Control</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #111827; }
    h1 { font-size: 1.4rem; }
    h2 { font-size: 1rem; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; margin: 1.5rem 0 .5rem; }
    .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; background: #fff; }
    .badge { display: inline-block; padding: .25rem .75rem; border-radius: 999px; color: #fff; font-weight: 600; }
    .stopped { background: #6b7280; }
    .starting { background: #d97706; }
    .running { background: #16a34a; }
    .unknown { background: #dc2626; }
    button { font-size: 1rem; padding: .6rem 1.2rem; margin-right: .5rem; border: 0; border-radius: 8px; cursor: pointer; color: #fff; }
    #start { background: #16a34a; } #stop { background: #dc2626; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    pre { background: #f3f4f6; padding: 1rem; border-radius: 8px; overflow: auto; font-size: .8rem; }
    .metric { display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; }
    .metric b { font-size: 1.2rem; }
    canvas { width: 100%; height: 120px; display: block; }
    .cams { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
    .cam { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #000; }
    .cam img { width: 100%; display: block; background: #111827; aspect-ratio: 16/9; object-fit: contain; }
    .cam .label { color: #e5e7eb; background: #111827; padding: .4rem .6rem; font-size: .85rem; display: flex; justify-content: space-between; }
    .dot { width: .6rem; height: .6rem; border-radius: 999px; display: inline-block; margin-right: .35rem; }
    .on { background: #16a34a; } .off { background: #6b7280; }
    .muted { color: #6b7280; font-size: .85rem; }
    .gpubar { height: .9rem; border-radius: 999px; background: #e5e7eb; overflow: hidden; }
    .gpubar > div { height: 100%; background: #2563eb; }
  </style>
</head>
<body>
  <h1>GR00T Inference Control</h1>

  <div class="grid">
    <div class="card">
      <h2>Deployment</h2>
      <p>Status: <span id="state" class="badge unknown">unknown</span></p>
      <p>
        <button id="start" onclick="act('start')">Start</button>
        <button id="stop" onclick="act('stop')">Stop</button>
      </p>
      <pre id="details">loading...</pre>
    </div>

    <div class="card">
      <h2>GPU allocation</h2>
      <div class="metric"><span id="gpuLabel">GPU</span><b id="gpuVal">--</b></div>
      <div class="gpubar"><div id="gpuFill" style="width:0%"></div></div>
      <div class="metric" id="gpuUtilRow" style="display:none;margin-top:.6rem"><span class="muted">utilization</span><b id="gpuUtilVal">--</b></div>
      <div class="gpubar" id="gpuUtilBarWrap" style="display:none"><div id="gpuUtilFill" style="width:0%;background:#d97706"></div></div>
      <p class="muted" id="gpuNote">Jetson Orin exposes no in-cluster utilization %; this shows GPUs held vs. capacity.</p>
    </div>

    <div class="card">
      <h2>CPU (inference pod)</h2>
      <div class="metric"><span class="muted">millicores</span><b id="cpuVal">--</b></div>
      <canvas id="cpuChart"></canvas>
    </div>

    <div class="card">
      <h2>Memory (inference pod)</h2>
      <div class="metric"><span class="muted">MiB</span><b id="memVal">--</b></div>
      <canvas id="memChart"></canvas>
    </div>
  </div>

  <h2>Camera streams</h2>
  <p class="muted" id="camNote">loading cameras…</p>
  <div class="cams" id="cams"></div>

  <script>
    const MAX_POINTS = 120;
    const cpuHist = [];
    const memHist = [];

    async function refresh() {
      try {
        const r = await fetch('/api/status');
        render(await r.json());
      } catch (e) { document.getElementById('details').textContent = 'Error: ' + e; }
    }
    async function act(which) {
      setButtons(true);
      try {
        const r = await fetch('/api/' + which, { method: 'POST' });
        render(await r.json());
      } catch (e) { document.getElementById('details').textContent = 'Error: ' + e; }
    }
    function setButtons(disabled) {
      document.getElementById('start').disabled = disabled;
      document.getElementById('stop').disabled = disabled;
    }
    function render(s) {
      const badge = document.getElementById('state');
      badge.textContent = s.state;
      badge.className = 'badge ' + s.state;
      document.getElementById('details').textContent = JSON.stringify(s, null, 2);
      document.getElementById('start').disabled = (s.state === 'running' || s.state === 'starting');
      document.getElementById('stop').disabled = (s.state === 'stopped');
    }

    function drawChart(id, data, color, fmt) {
      const c = document.getElementById(id);
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      c.width = w * dpr; c.height = h * dpr;
      const ctx = c.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);
      if (!data.length) return;
      const max = Math.max(1, ...data);
      const pad = 4;
      // gridlines
      ctx.strokeStyle = '#f3f4f6'; ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const y = pad + (h - 2 * pad) * i / 4;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      // line + fill
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = (w * i) / Math.max(1, MAX_POINTS - 1);
        const y = pad + (h - 2 * pad) * (1 - v / max);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
      ctx.lineTo((w * (data.length - 1)) / Math.max(1, MAX_POINTS - 1), h);
      ctx.lineTo(0, h); ctx.closePath();
      ctx.fillStyle = color + '22'; ctx.fill();
    }

    async function pollMetrics() {
      try {
        const r = await fetch('/api/metrics');
        const m = await r.json();
        const cpu = m.totalCpuMillicores || 0;
        const memMiB = Math.round((m.totalMemBytes || 0) / (1024 * 1024));
        const running = (m.pods && m.pods.length > 0);
        cpuHist.push(running ? cpu : 0); if (cpuHist.length > MAX_POINTS) cpuHist.shift();
        memHist.push(running ? memMiB : 0); if (memHist.length > MAX_POINTS) memHist.shift();
        document.getElementById('cpuVal').textContent = running ? cpu : '—';
        document.getElementById('memVal').textContent = running ? memMiB : '—';
        drawChart('cpuChart', cpuHist, '#2563eb');
        drawChart('memChart', memHist, '#16a34a');
      } catch (e) { /* keep last */ }
    }

    async function pollGpu() {
      try {
        const r = await fetch('/api/gpu');
        const g = await r.json();
        document.getElementById('gpuLabel').textContent = g.resourceName || 'GPU';
        document.getElementById('gpuVal').textContent = g.inUse + ' / ' + g.capacity + ' in use';
        const pct = g.capacity ? Math.min(100, (g.inUse / g.capacity) * 100) : 0;
        document.getElementById('gpuFill').style.width = pct + '%';
        const hasUtil = (g.utilizationPercent !== null && g.utilizationPercent !== undefined);
        document.getElementById('gpuUtilRow').style.display = hasUtil ? 'flex' : 'none';
        document.getElementById('gpuUtilBarWrap').style.display = hasUtil ? 'block' : 'none';
        if (hasUtil) {
          document.getElementById('gpuUtilVal').textContent = g.utilizationPercent + '%';
          document.getElementById('gpuUtilFill').style.width = Math.min(100, g.utilizationPercent) + '%';
          document.getElementById('gpuNote').textContent = 'Live GPU load from the Jetson sysfs counter, plus GPUs held vs. capacity.';
        }
      } catch (e) { /* keep last */ }
    }

    async function loadCameras() {
      try {
        const r = await fetch('/api/cameras');
        const data = await r.json();
        const cams = data.cameras || [];
        const note = document.getElementById('camNote');
        if (data.error) { note.textContent = 'Camera streamer unreachable: ' + data.error; return; }
        if (!cams.length) { note.textContent = 'No cameras reported by the streamer.'; return; }
        note.textContent = cams.length + ' camera(s) — live MJPEG, proxied through this UI.';
        const wrap = document.getElementById('cams');
        wrap.innerHTML = '';
        cams.forEach(cam => {
          const div = document.createElement('div');
          div.className = 'cam';
          const dot = cam.connected ? 'on' : 'off';
          div.innerHTML =
            '<img alt="' + cam.name + '" src="' + cam.stream_url + '" ' +
            'onerror="this.src=\\'' + cam.snapshot_url + '\\'" />' +
            '<div class="label"><span><span class="dot ' + dot + '"></span>' +
            (cam.name || cam.id) + '</span><span>' + (cam.resolution || '') + '</span></div>';
          wrap.appendChild(div);
        });
      } catch (e) {
        document.getElementById('camNote').textContent = 'Failed to load cameras: ' + e;
      }
    }

    refresh(); pollMetrics(); pollGpu(); loadCameras();
    setInterval(refresh, 3000);
    setInterval(pollMetrics, 3000);
    setInterval(pollGpu, 5000);
    setInterval(loadCameras, 30000);
  </script>
</body>
</html>
"""
