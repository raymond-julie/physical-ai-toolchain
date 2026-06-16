---
title: Using Camera Streamer and the URCap Pendant Viewer
description: End-to-end tutorial for publishing Orbbec camera feeds with camera_streamer and viewing them on a Universal Robots teach pendant with the Camera Stream URCap
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: tutorial
keywords:
  - camera streamer
  - urcap
  - pendant viewer
  - orbbec
  - mjpeg
estimated_reading_time: 18
---

This tutorial walks through the two projects that make the live camera experience work on the edge rig: `camera_streamer`, which publishes MJPEG feeds on the network, and the Java URCap, which renders one of those feeds directly on a PolyScope 5 pendant. Use this when you want a human operator to see the rig cameras without opening a browser on a separate workstation.

## 📋 Prerequisites

| Requirement | Details |
| --- | --- |
| Edge host | Linux host with the cameras physically attached or reachable through the host's `/dev` V4L2 devices |
| Cameras | Orbbec devices supported by `pyorbbecsdk`, or synthetic fallback for functional testing |
| Python | Python 3.10+ for `camera_streamer` |
| UR controller | PolyScope 5 e-Series controller for the URCap |
| Java toolchain | JDK 8 and Maven 3 for building the URCap |
| URCap SDK | Universal Robots URCap SDK unpacked locally |
| Network plan | Replace all `192.168.1.x` placeholders with the actual robot or workstation IPs in your environment |

## 🚀 Step 1: Start the camera streamer locally

Move into the project directory and install the Python dependencies.

```bash
cd data-pipeline/capture/camera_streamer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the Orbbec SDK is already installed on the host, start the streamer with auto-discovery.

```bash
./run_streamer.sh
```

If the rig uses a TrainMyBot device manifest, point the streamer at that config so it serves only the declared camera devices.

```bash
./run_streamer.sh --config /etc/trainmybot/config_v3.yaml
```

If the host has no SDK or no physical cameras attached, keep going anyway. The streamer falls back to synthetic frames, which is enough to validate URLs, browser rendering, and the URCap integration.

## 🔎 Step 2: Verify the streamer endpoints from a browser

Open the dashboard the process prints. A typical example looks like `http://192.168.1.x:8000`, but you must replace that placeholder with the actual IP of the edge host.

Verify these endpoints in order:

| Endpoint | Expected result |
| --- | --- |
| `/` | HTML dashboard with one card per camera |
| `/api/cameras` | JSON camera catalog with `id`, `stream_url`, and `snapshot_url` |
| `/snapshot/<camera_id>` | One JPEG frame |
| `/stream/<camera_id>` | Continuous MJPEG stream |
| `/healthz` | Simple liveness JSON |

Quick CLI check:

```bash
curl http://192.168.1.x:8000/api/cameras
curl http://192.168.1.x:8000/healthz
```

Replace `192.168.1.x` with the real streamer host IP before running those commands.

## 📡 Step 3: Confirm the stream is consumable by other clients

Before you bring the URCap into the loop, prove that a normal client can decode the stream.

Browser test:

```text
http://<streamer-host>:8000/stream/<camera-id>
```

VLC test:

```text
Media -> Open Network Stream -> http://<streamer-host>:8000/stream/<camera-id>
```

OpenCV test:

```python
import cv2

cap = cv2.VideoCapture('http://<streamer-host>:8000/stream/<camera-id>')
ok, frame = cap.read()
print(ok, None if frame is None else frame.shape)
```

Do not continue to the URCap build until at least one of those tests works. The URCap is only an MJPEG client. If the browser cannot decode the stream, the pendant cannot either.

## 🐳 Step 4: Optional container deployment for the streamer

Use this path when the rig standardizes on Docker rather than a host Python environment.

Stage the Orbbec SDK payload into the Docker build context first:

```bash
cd data-pipeline/capture/camera_streamer
bash docker/stage-orbbec.sh
docker compose build
docker compose up -d
```

Verify the container is reachable on the host network:

```bash
docker ps
curl http://192.168.1.x:8000/healthz
```

Again, replace the placeholder with the actual streamer host IP.

## 🔧 Step 5: Build the Camera Stream URCap

The URCap is a separate Java project. Build it only after the streamer side is working.

Install the Java toolchain:

```bash
sudo apt-get install -y openjdk-8-jdk maven
```

Then build the URCap from its project directory.

```bash
cd data-pipeline/capture/ur/urcap
URCAP_SDK_DIR=/path/to/unpacked/sdk ./build.sh
```

Expected result:

```text
com.trainmybot.camerastream.impl/target/com.trainmybot.camerastream.impl-<version>.urcap
```

If the build fails because `com.ur.urcap:api` cannot be resolved, the SDK artifacts are not installed in your Maven cache yet. Fix that before debugging anything in the code.

## 💾 Step 6: Install the URCap on the robot

Copy the generated `.urcap` file to removable media and install it on the pendant.

1. Insert the USB stick into the controller.
2. Open `Settings -> System -> URCaps`.
3. Press `+` and select the `.urcap` file.
4. Restart PolyScope when prompted.
5. Open `Installation -> URCaps -> Camera Stream`.

At this point the installation node should exist, even if it is not connected to a real stream yet.

## 🖥️ Step 7: Configure the streamer URL and camera id on the pendant

The URCap stores two pieces of information:

| Field | Meaning |
| --- | --- |
| Streamer URL | Base URL such as `http://192.168.1.x:8000` |
| Camera id | The camera serial or other id exposed by `/api/cameras` |

On the pendant:

1. Tap the `Streamer URL` field.
2. Enter the real streamer host URL. Do not leave `192.168.1.x` as a placeholder.
3. Tap the `Camera id` field.
4. Paste one of the ids reported by `http://<streamer-host>:8000/api/cameras`.
5. Press `Reconnect`.

If the feed appears in the Installation screen, the end-to-end chain is working.

## 🧭 Step 8: Use the toolbar popup during operation

The URCap also registers a toolbar contribution. That matters because operators do not stay on the Installation screen while running the cell.

Use it like this:

1. Leave the configuration saved in the Installation node.
2. Navigate to any normal PolyScope screen.
3. Open the camera toolbar popup from the header bar.
4. Confirm the same live MJPEG feed appears there.

If the Installation screen works but the toolbar does not, the issue is usually in the OSGi registration or the persisted installation data lookup, not in the streamer.

## ✅ Step 9: Operational checks before handing the rig to an operator

Run this checklist:

| Check | Expected result |
| --- | --- |
| Browser stream | Live MJPEG feed on a workstation browser |
| Pendant installation node | Same feed rendered inside PolyScope |
| Pendant toolbar popup | Same feed rendered outside the installation page |
| Stream reconnect | Brief network interruption recovers automatically |
| Camera id switch | Changing the camera id swaps the visible feed |

## 🔍 Troubleshooting

### Pendant shows a blank panel

Check the streamer URL first. If the pendant is still pointed at a placeholder like `192.168.1.x`, it is not a code problem.

### Browser works, URCap does not

Confirm the robot controller can route to the streamer host. The controller network is often isolated from the developer workstation VLAN.

### URCap installs but does not appear in PolyScope

Verify the build used Java 8 and a compatible URCap API version.

### Stream reconnects forever

Open `http://<streamer-host>:8000/api/cameras` and confirm the selected camera id still exists. If the streamer restarted with a different device set, the saved id may now be invalid.

## 🔗 Related documentation

- [camera_streamer README](../../../data-pipeline/capture/camera_streamer/README.md)
- [Camera Stream URCap README](../../../data-pipeline/capture/ur/urcap/README.md)
- [Recording with the Dual Recorder](using-dual-recorder.md)
