---
title: Operating Edge Capture Projects
description: End-to-end tutorial for using the new edge capture projects, including camera_streamer, the URCap, dual_recorder, episode_recorder, leader_follower, and dataset tools
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: tutorial
keywords:
  - edge capture
  - ur recorder
  - camera streamer
  - dual recorder
  - episode recorder
  - leader follower
  - lerobot
estimated_reading_time: 35
sidebar_position: 3
---

Use the new edge capture projects as a complete data-collection toolbox, not as isolated scripts. This tutorial explains when to use each project, how they fit together, and how to run them safely on a real robot cell.

## 📋 Prerequisites

| Requirement | Details |
| --- | --- |
| Linux host | Ubuntu 22.04 or 24.04 recommended |
| Python | 3.10 or later |
| Robot network | All example placeholders such as `192.168.1.x` must be replaced with the actual robot, camera, or workstation IPs for your cell |
| UR robots | Remote access enabled when using RTDE-based tools |
| Cameras | Orbbec cameras for `camera_streamer` and `dual_recorder`; RealSense or Nova camera streams for `episode_recorder` and `leader_follower` |
| Storage | Enough local disk for LeRobot datasets and encoded MP4 files |

## 🎯 Choose the Right Project

Use this decision table before you start:

| Goal | Project | Use it when |
| --- | --- | --- |
| Share live camera feeds on the network | `data-pipeline/capture/camera_streamer` | You need browser, VLC, OpenCV, or URCap access to Orbbec cameras |
| Show camera feeds on the UR teach pendant | `data-pipeline/capture/ur/urcap` | Operators need the live stream directly inside PolyScope |
| Record two follower arms plus multiple Orbbec cameras | `data-pipeline/capture/ur/dual_recorder` | You want read-only LeRobot capture from a dual-arm UR cell |
| Record N robots and N cameras with pluggable drivers | `data-pipeline/capture/ur/episode_recorder` | You want a configurable, vendor-agnostic recording stack |
| Mirror one UR arm to another while recording | `data-pipeline/capture/ur/leader_follower` | You want teleoperation plus dataset capture |
| Convert or merge datasets after capture | `data-management/tools` | You need LeRobot conversion, MCAP conversion, or session merges |

## 🚀 Tutorial 1: Start the Camera Streamer

Start here when anything else in the stack needs Orbbec video. The streamer is the simplest component and acts as the shared camera source for both humans and other software.

### Step 1: Install and launch it

```bash
cd data-pipeline/capture/camera_streamer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_streamer.sh
```

If you use a TrainMyBot camera config, restrict the exposed cameras to that file:

```bash
./run_streamer.sh --config /etc/trainmybot/config_v3.yaml
```

### Step 2: Verify the dashboard and raw endpoints

Open the printed dashboard URL from another machine on the same LAN. Then verify the low-level endpoints directly:

```bash
curl http://<streamer-host>:8000/api/cameras
curl http://<streamer-host>:8000/healthz
```

Test one stream in a browser or VLC:

```text
http://<streamer-host>:8000/stream/<camera-id>
```

### Step 3: Decide whether this stays as a standalone service

Keep `camera_streamer` running when:

1. Multiple viewers need the same physical cameras.
2. You want the URCap to show the feed on the pendant.
3. Another recorder should consume MJPEG over HTTP instead of opening the devices directly.

Do not run it when a recorder needs exclusive direct access to the same cameras and has no HTTP-stream option.

> [!WARNING]
> The streamer binds to `0.0.0.0` and has no built-in authentication. Run it only on a trusted network, or place it behind access control.

## 🎥 Tutorial 2: Put the Camera Feed on the Teach Pendant with the URCap

Use the URCap after the streamer works. The URCap does not talk to the cameras directly. It only renders the streamer's MJPEG output inside PolyScope.

### Step 1: Build the URCap

```bash
cd data-pipeline/capture/ur/urcap
URCAP_SDK_DIR=/path/to/unpacked/urcap-sdk ./build.sh
```

The output is a `.urcap` bundle in the module `target/` directory.

### Step 2: Install it on the robot controller

1. Copy the generated `.urcap` file to a USB stick.
2. Insert the USB stick into the teach pendant controller.
3. Open `Settings -> System -> URCaps`.
4. Add the URCap and restart PolyScope.

### Step 3: Point it at the streamer

In `Installation -> URCaps -> Camera Stream`:

1. Set the streamer URL to `http://<streamer-host>:8000`.
2. Set the camera id to the serial or logical stream id reported by `/api/cameras`.
3. Press `Reconnect`.

If the preview does not update:

1. Confirm the controller can reach the streamer host.
2. Confirm you did not leave `127.0.0.1` in the URL field.
3. Confirm the selected camera id exists in the streamer dashboard.

## 🦾 Tutorial 3: Record a Dual-Arm UR Cell with `dual_recorder`

Use `dual_recorder` when you want a dedicated read-only UR capture app for a dual-arm follower setup with Orbbec cameras.

### Step 1: Prepare the cell configuration

`dual_recorder` expects the TrainMyBot device topology in `/etc/trainmybot/config_v3.yaml`. Before launching, confirm that file contains:

1. The left and right follower arms.
2. The Orbbec camera serials.
3. The correct robot IPs for your cell.

The example placeholders in the repo use `192.168.1.x`. Replace them with the real robot IPs before you copy any example config into production.

### Step 2: Install and run the recorder

```bash
cd data-pipeline/capture/ur/dual_recorder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_dual_recorder.sh
```

Useful variants:

```bash
./run_dual_recorder.sh --no-web
./run_dual_recorder.sh --no-record
./run_dual_recorder.sh --no-di0-trigger
```

### Step 3: Understand the runtime flow

While it runs:

1. `arm_reader.py` reads both follower arms over RTDE.
2. `robotiq.py` reads each gripper position over the socket interface.
3. `cameras.py` captures the configured Orbbec streams.
4. `recorder.py` assembles LeRobot frames.
5. The Flask dashboard exposes state, recording toggles, and previews.

### Step 4: Verify the output dataset

After stopping a recording, inspect the generated session:

```bash
find recordings_lerobot -maxdepth 3 -type f | head -50
```

You should see:

1. `meta/info.json`
2. `meta/episodes.jsonl`
3. `data/chunk-000/episode_*.parquet`
4. `videos/chunk-000/.../episode_*.mp4`

## 🧰 Tutorial 4: Record a Configurable Cell with `episode_recorder`

Use `episode_recorder` when the UR-only, dual-arm assumptions of `dual_recorder` are too narrow. This is the flexible option for pluggable robot and gripper drivers.

### Step 1: Install prerequisites

```bash
cd data-pipeline/capture/ur/episode_recorder
./install_dependencies.sh
source /opt/ros/${ROS_DISTRO}/setup.bash
```

### Step 2: Start with the defaults

```bash
./run_episode_recorder.sh
```

This launches:

1. One or more camera sources.
2. `robot_reader` for each configured robot.
3. `trigger_tool_io` for a physical trigger.
4. `trigger_gui` for a web trigger.
5. `episode_recorder` for LeRobot output.

### Step 3: Switch modes deliberately

Examples:

```bash
./run_episode_recorder.sh --gripper2-driver none
./run_episode_recorder.sh --depth --camera-fps 30
./run_episode_recorder.sh --no-tool-trigger
./run_episode_recorder.sh --state-source nova
```

Use `--state-source nova` when Nova owns the RTDE lock and you want the recorder to subscribe to the NATS controller-state stream instead of talking to the controller directly.

### Step 4: Add a new robot driver

This is the main extension story for `episode_recorder`:

1. Create `episode_recorder/drivers/<vendor>.py`.
2. Implement `RobotStateDriver`.
3. Optionally implement `GripperDriver`.
4. Register both in `registry.py`.
5. Launch with `--robot1-driver <vendor>`.

That extension path is the reason to choose this project instead of cloning and modifying `leader_follower` or `dual_recorder`.

## 🤝 Tutorial 5: Run `leader_follower` for Teleoperation Plus Recording

Use `leader_follower` only when you need one robot to mirror another in real time. Unlike the recorder-only projects, this stack can physically move the follower arm.

### Step 1: Start in dry-run mode

```bash
cd data-pipeline/capture/ur/leader_follower
./install_dependencies.sh
source /opt/ros/${ROS_DISTRO}/setup.bash
./run_recorder.sh --no-motion
```

Dry run lets you verify:

1. RTDE connectivity.
2. Camera feeds.
3. Dashboard state changes.
4. Dataset writing.

It does not move the follower arm.

### Step 2: Test the trigger and dataset path

Use either:

1. The physical DI0 button on the leader tool.
2. The dashboard start button.

Start and stop one episode. Confirm that `recordings_lerobot` now contains a new session with parquet and MP4 files.

### Step 3: Enable real motion only after validation

```bash
./run_recorder.sh
```

Or align to the configured home pose at the beginning and end of each cycle:

```bash
./run_recorder.sh --home
```

### Step 4: Understand when not to use it

Do not use `leader_follower` when:

1. You only need read-only data capture.
2. You need more than one robot pair.
3. You need a driver-pluggable architecture.

In those cases, use `dual_recorder` or `episode_recorder` instead.

## 🔄 Tutorial 6: Convert and Merge Datasets After Capture

Once capture finishes, the new tools in `data-management/tools` help normalize the output for training.

### Convert LeRobot v3 to the GR00T-flavored v2 layout

```bash
cd data-management/tools
python -m pip install -r requirements.txt
python convert_lerobot_v3_to_v2.py --src /data/run-v3 --dst /data/run-v2
```

Use this when a training or evaluation flow still expects the older per-episode v2 structure.

### Merge multiple `session_*` datasets

```bash
python merge_lerobot_sessions.py ./downloads ./combined-sessions
```

Use this when an operator collected several short sessions and you want one training-ready dataset.

### Convert MCAP recordings to LeRobot

```bash
cd lerobot-converter
python -m pip install -r requirements.txt
python convert_mcap_to_lerobot.py \
  --src /home/aloha/recordings \
  --out /data/pick-place-v1 \
  --repo-id pick-place-v1 \
  --task "pick up the block and place it in the bin" \
  --overwrite
```

Use the inspection tools before converting a new recording source:

```bash
python inspect_mcap.py /path/to/file.mcap
python inspect_messages.py /path/to/file.mcap
python check_motion.py /path/to/file.mcap
```

## ✅ Verification Checklist

You have the stack working when all of the following are true:

1. The selected recorder writes `meta/`, `data/`, and `videos/` outputs.
2. The camera streamer dashboard shows every expected camera.
3. The URCap shows a live feed on the pendant when configured.
4. Trigger start and stop actions create complete episodes.
5. Dataset conversion or merge commands succeed on the produced sessions.

## 🔗 Related Documentation

- [Configuring Edge Data Recording](configuring-edge-data-recording.md)
- [Preparing Datasets for Training](preparing-datasets-for-training.md)
- [Your First LeRobot Training Job](../training/your-first-lerobot-training-job.md)
