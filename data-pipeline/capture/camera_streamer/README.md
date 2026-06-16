# camera_streamer

Host every connected camera as a shareable MJPEG link so anyone on the local network can consume the stream — in a browser, VLC, OpenCV, or any other MJPEG-capable client. One capture thread per camera feeds an unlimited number of viewers, so extra consumers add no device load.

## 📋 Prerequisites

- Python 3.10+
- `flask`, `opencv-python-headless`, `numpy`, `PyYAML` (see [requirements.txt](requirements.txt))
- Orbbec SDK (`pyorbbecsdk`) for live cameras — optional; a synthetic test pattern is served when it is absent

## 🚀 Quick Start

```bash
cd data-pipeline/capture/camera_streamer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Auto-discover all connected Orbbec cameras and serve on :8000
./run_streamer.sh

# Or pick a port / select cameras from a device config
./run_streamer.sh --port 9000
./run_streamer.sh --config /etc/trainmybot/config_v3.yaml
```

Open the printed dashboard URL (e.g. `http://192.168.1.20:8000`) from any machine on the same network. The dashboard exposes a live thumbnail and copyable share link per camera; per-camera endpoints are `/stream/<id>`, `/snapshot/<id>`, `/api/cameras`, and `/healthz`.

> [!WARNING]
> Streams bind on `0.0.0.0` and have no authentication. Run only on a trusted
> network, or place the service behind a reverse proxy for access control.

## 📡 Consuming a Stream

Each camera link is a standard `multipart/x-mixed-replace` MJPEG stream:

- Browser: open `http://<host>:8000/stream/<camera_id>` directly.
- VLC: Media → Open Network Stream → paste the stream URL.
- OpenCV: `cv2.VideoCapture("http://<host>:8000/stream/<camera_id>")`.
- ffmpeg: `ffmpeg -i http://<host>:8000/stream/<camera_id> out.mp4`.

## ⚙️ CLI Options

| Flag | Description |
| --- | --- |
| `--config <path>` | Device `config_v3.yaml`; serve only its `type: camera` devices. Omit to auto-discover. |
| `--app-config <path>` | Overlay YAML for server/stream settings (see [config/app.yaml](config/app.yaml)). |
| `--host <addr>` | Bind address (default `0.0.0.0`). |
| `--port <n>` | Bind port (default `8000`). |
| `--quality <1-100>` | JPEG quality (default `80`). |
| `--fps <n>` | Cap on streamed frame rate (default `15`). |
| `--max-width <px>` | Downscale frames wider than this before encoding (`0` = off). |
| `--list` | List the cameras that would be served, then exit. |
| `-v`, `--verbose` | Debug logging. |

## 📷 Cameras (Orbbec)

The Orbbec SDK (`pyorbbecsdk`) is not on PyPI and must be built from source for your platform. On the lab Jetson it is already built and installed user-level.

> [!NOTE]
> If the SDK is missing or a device fails to open, the affected camera falls
> back to a labelled synthetic test pattern so the service still runs end-to-end.

Gemini 305g cameras are stereo and expose `LEFT_COLOR_SENSOR` / `RIGHT_COLOR_SENSOR` rather than a single `COLOR_SENSOR`; the capture code handles this automatically. The same physical camera cannot be opened by two processes at once — stop any recorder using the device before streaming it.

## 🐳 Container

`pyorbbecsdk` is a from-source build pinned to this platform (aarch64 / CPython 3.10 / Ubuntu 22.04), so the image reuses the host's prebuilt SDK instead of recompiling it. GMSL kernel modules stay on the host; the container only needs the `/dev/video*` device nodes passed through.

```bash
cd data-pipeline/capture/camera_streamer
bash docker/stage-orbbec.sh   # stage the prebuilt SDK into the build context
docker compose build
docker compose up -d
```

> [!NOTE]
> Cameras must be enumerated on the host (load the GMSL driver) **before**
> starting the container; the container does not load kernel modules.
> `network_mode: host` keeps LAN streams reachable and the printed share URLs
> correct. `build_and_push.sh <tag>` builds and pushes to an Azure Container
> Registry (override `REGISTRY`, `IMAGE`).

## 🧪 Tests

```bash
cd data-pipeline/capture/camera_streamer
python -m pytest tests
```

Tests run without hardware — `pyorbbecsdk` is imported lazily and absent in the test environment.
