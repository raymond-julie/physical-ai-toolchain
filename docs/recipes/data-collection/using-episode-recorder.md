---
title: Recording Robot Datasets with the Vendor-Agnostic Episode Recorder
description: End-to-end tutorial for configuring, launching, and validating the multi-robot episode_recorder stack in RTDE and Nova modes
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: tutorial
keywords:
  - episode recorder
  - lerobot
  - nova
  - rtde
  - multi-robot
estimated_reading_time: 24
---

This tutorial shows how to use `episode_recorder`, the vendor-agnostic recorder that turns robot state and camera topics into LeRobot episodes. Use it when you want a configurable capture stack that can switch between RTDE and Nova, local RealSense and Nova WebRTC cameras, and multiple robot or gripper drivers without rewriting the recorder.

## 📋 Prerequisites

| Requirement | Details |
| --- | --- |
| ROS 2 | Humble or Jazzy installed and sourced |
| Python | Python 3.10+ |
| Cameras | RealSense locally, or Nova camera-manager if using WebRTC bridging |
| Robot connectivity | RTDE access to robots, or Nova NATS access if using `--state-source nova` |
| Config files | `config/robots.example.yaml` as a starting point and, for Nova, `nova.example.env` |
| Network plan | Replace all `192.168.1.x` placeholders with the real IPs for the rig before running |

## 📦 Step 1: Install the stack once

```bash
cd data-pipeline/capture/ur/episode_recorder
./install_dependencies.sh
source /opt/ros/${ROS_DISTRO}/setup.bash
```

Do not skip the installer on a fresh machine. The project depends on a mix of apt packages, ROS packages, and pip packages.

## 🧭 Step 2: Pick the operating mode before you launch

The recorder has two major dimensions of configuration.

| Dimension | Option A | Option B |
| --- | --- | --- |
| Robot state source | `rtde` | `nova` |
| Camera source | `local` | `nova` |

Recommended combinations:

| Use case | Suggested flags |
| --- | --- |
| Local development on a lab workstation | `--state-source rtde --camera-source local` |
| Running alongside Nova without RTDE contention | `--state-source nova --camera-source nova` |
| Hybrid migration | `--state-source nova --camera-source local` |

Choose the mode first. The rest of the setup depends on that choice.

## ⚙️ Step 3: Start from the example robot config

Copy the sample config and edit it for the real cell.

```bash
cp config/robots.example.yaml config/robots.local.yaml
```

Update at least these fields:

| Field | Meaning |
| --- | --- |
| `robots[].name` | ROS namespace prefix |
| `robots[].ip` | Actual robot controller IP, not the `192.168.1.x` placeholder |
| `robot_driver` | For example `ur_rtde` or `nova` |
| `gripper_driver` | For example `robotiq_socket` or `none` |
| `trigger` | Which robot's DI0 acts as the record toggle |

If you use Nova, also copy and edit the environment template.

```bash
cp nova.example.env nova.env
```

Populate the real NATS URL, camera API base URL, and any credentials required by the Nova deployment.

## ▶️ Step 4: Launch the simplest working configuration first

Start with the local RTDE and local camera path.

```bash
./run_episode_recorder.sh
```

This default path gives you:

| Component | Behavior |
| --- | --- |
| Robot readers | Two UR RTDE readers with Robotiq state |
| Cameras | Local `realsense2_camera` nodes |
| Trigger | Physical DI0 plus GUI fallback |
| Recorder | LeRobot dataset writer |
| UI | Browser control page on port `8080` |

Only move to Nova mode after this baseline works.

## 🧪 Step 5: Validate the trigger path and episode write path

Use one short test episode.

1. Open the GUI at `http://<host>:8080`.
2. Press the physical DI0 or the Record button.
3. Let it run for 10-15 seconds.
4. Stop the episode.
5. Inspect the generated output under `recordings_lerobot`.

You want to see a complete episode directory, not just partial metadata.

## 🛰️ Step 6: Switch to Nova mode when RTDE is not the right transport

Once the local path works, move to Nova mode deliberately.

Minimal launch:

```bash
./run_episode_recorder.sh --state-source nova
```

More explicit launch:

```bash
./run_episode_recorder.sh \
  --state-source nova \
  --nova-nats-url nats://192.168.1.x:31422 \
  --nova-cell cell \
  --nova-ctrl1 ur5-left \
  --nova-ctrl2 ur5-right
```

Important behavior differences:

| Change | Effect |
| --- | --- |
| Nova state stream | Robot positions come from NATS instead of RTDE |
| Tool DI in Nova | Not available, so the launcher auto-disables the tool trigger |
| Gripper state | Still read directly from the robots |

In Nova mode, use the GUI button as the main trigger unless you have an alternate trigger design.

## 🎥 Step 7: Switch cameras to Nova WebRTC if the cluster owns the sensors

If the cameras are managed by Nova instead of the local workstation, launch with the Nova camera bridge.

```bash
./run_episode_recorder.sh --state-source nova --camera-source nova
```

The bridge does three things:

1. Discovers camera devices from the Nova API.
2. Performs the server-offer WebRTC handshake.
3. Republishes each color stream as a ROS image topic that matches the local RealSense topic shape.

That design means the actual recorder code does not care whether the images came from USB or Nova.

## 🧩 Step 8: Add a new robot or gripper vendor without rewriting the recorder

This is one of the main reasons the project exists.

To add a new vendor:

1. Create a new driver module under `episode_recorder/drivers/`.
2. Implement the base driver interface.
3. Register it in the driver registry.
4. Point the launcher at the new driver name.

The capture pipeline stays the same. Only the vendor adapter changes.

## ✅ Step 9: Define the steady-state operator workflow

Use this order for real runs:

1. Source ROS and verify environment variables.
2. Start the launcher in the intended mode.
3. Confirm the UI is reachable.
4. Confirm robot state topics are live.
5. Confirm camera topics are live.
6. Record one short validation episode.
7. Inspect the output directory.
8. Run the real capture session.

That sequence avoids discovering configuration problems after hours of collection.

## 🔍 Troubleshooting

### Recorder starts but writes no usable episodes

Confirm all declared robot and camera topics have actually published at least one message. The recorder intentionally refuses to frame incomplete observations.

### Nova mode produces no robot motion data

The controller-state stream only exists while a Nova motion-group session is active. Use the provided keep-alive helper if necessary.

### GUI works but physical trigger does nothing

That is expected in Nova mode. Tool DI is not present there.

### One robot has no gripper

Set that robot's `gripper_driver` to `none` instead of working around it in the recorder logic.

## 🔗 Related documentation

- [episode_recorder README](../../../data-pipeline/capture/ur/episode_recorder/README.md)
- [Recording Dual-Arm UR Data with the Dual Recorder](using-dual-recorder.md)
- [Teleoperating and recording with the leader/follower stack](using-leader-follower-teleop.md)
