# GR00T Control UI

A minimal FastAPI control + observability UI for the GR00T inference
deployment. It gives an operator a single page to **start/stop** the policy
server and watch how a run is going, without `kubectl`.

## 🎛️ What it does

| Panel               | Behavior                                                                                                              |
|---------------------|---------------------------------------------------------------------------------------------------------------------|
| Start / Stop / Status | "Start" scales the target Deployment to 1, "Stop" scales it to 0 (freeing the GPU; cached weights stay on the PVC). |
| Camera streams      | Reverse-proxies the `ur-camera-streamer` MJPEG feeds so one port-forward surfaces every Orbbec camera.               |
| CPU / memory graphs | Inference pod CPU and memory over time, read from the metrics-server (`metrics.k8s.io`).                             |
| GPU allocation      | Best-effort node GPU capacity vs. in-use; live utilization when the Jetson sysfs load file is mounted.              |

It does **not** drive the robots — that is the
[GR00T robot client](../gr00t-robot-client). The UI only scales the policy
server up and down.

## 🔒 Security model

- Runs in-cluster on the mounted ServiceAccount token. RBAC restricts it to
  get/scale on the single target Deployment plus read-only pods, pod-metrics,
  and nodes (chart `control-ui-rbac.yaml`).
- Camera ids are constrained by `^[A-Za-z0-9_.-]+$` before being used in proxied
  paths, preventing SSRF / path traversal through the camera proxy.

## ⚙️ Configuration

All via environment variables (the chart injects them):

| Variable             | Default                  | Purpose                                                     |
|----------------------|--------------------------|-------------------------------------------------------------|
| `TARGET_NAMESPACE`   | `default`                | Namespace of the inference Deployment.                      |
| `TARGET_DEPLOYMENT`  | `gr00t-gr00t-inference`  | Deployment to scale and watch.                              |
| `CAMERA_STREAMER_URL`| `http://127.0.0.1:8000`  | Base URL of the `ur-camera-streamer` (downward-API HOST_IP).|
| `GPU_RESOURCE_NAME`  | `nvidia.com/gpu`         | GPU resource name advertised by the device plugin.          |
| `GPU_LOAD_FILE`      | _(unset)_                | Optional Jetson sysfs GPU load file for live utilization.   |

## 🚀 Run it

Locally (uses your kubeconfig when not in-cluster):

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

Build and push the image, then enable it in the chart (`controlUi.enabled`):

```bash
./build_and_push.sh 0.1
```

The registry is parameterized: `build_and_push.sh` reads `${REGISTRY}` (default
`immitationlearning.azurecr.io`) so the image can target any registry.
