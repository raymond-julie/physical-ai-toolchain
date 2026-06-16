---
title: Teleoperating and Recording with the Leader-Follower Stack
description: End-to-end tutorial for running the leader_follower stack in dry-run and execute modes, validating the dashboard, and recording teleoperation episodes safely
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: tutorial
keywords:
  - leader follower
  - teleoperation
  - lerobot
  - ur5e
  - edge capture
estimated_reading_time: 20
---

This tutorial shows how to use `leader_follower`, the teleoperation stack that mirrors a manually moved leader UR arm onto a follower arm while recording the session as a LeRobot dataset. Use it when you need paired state and action data from a real teleop session, not just passive observation.

## 📋 Prerequisites

| Requirement | Details |
| --- | --- |
| Robots | Reachable leader and follower UR robots |
| Cameras | RealSense devices, recorded videos, or synthetic camera fallback |
| ROS 2 | Humble or Jazzy |
| Python | Python 3.10+ |
| Safety prep | Clear workspace and accessible e-stop |
| Network plan | Replace every `192.168.1.x` placeholder with the actual robot or workstation IPs before running |

## 🛠️ Step 1: Install dependencies and source ROS

```bash
cd data-pipeline/capture/ur/leader_follower
./install_dependencies.sh
source /opt/ros/${ROS_DISTRO}/setup.bash
```

If you are on a shared lab machine, validate that `ur_rtde`, `lerobot`, and `cv_bridge` all import cleanly before you continue.

## 🧪 Step 2: Start in dry-run mode first

Never start with live robot motion unless you have already validated the entire stack on that machine.

```bash
./run_recorder.sh --no-motion
```

This mode still starts the stack, UI, and recorder, but it does not command the follower.

Validate these pieces:

| Component | Expected result |
| --- | --- |
| Source reader | Leader joints and gripper update in the dashboard |
| Destination writer | State machine updates appear in the dashboard |
| Recorder | Toggle path changes `/recorder/active` |
| Cameras | Live previews appear in the GUI |

## 🎥 Step 3: Use simulated cameras when RealSense hardware is unavailable

If the rig cameras are unavailable, do not block the whole test session. Use the built-in replay path.

Replay one video:

```bash
./run_recorder.sh --no-motion --video path/to/cam1.mp4
```

Replay two videos:

```bash
./run_recorder.sh --no-motion --video cam1.mp4 --video2 cam2.mp4
```

Use the synthetic pattern when you do not even have a video file:

```bash
./run_recorder.sh --no-motion --synthetic-camera
```

This is the fastest way to prove the recorder and GUI still work when hardware is missing.

## 🧭 Step 4: Decide whether to use the home-position workflow

The stack supports two operator styles:

| Mode | Behavior |
| --- | --- |
| `--home` | Follower aligns to a home pose on start and returns there after stop |
| `--no-home` | Follower mirrors directly from its current pose |

Use `--home` when you want repeatable session starts. Use direct mirroring when you already have the rigs aligned manually and want the shortest path to motion.

## 🚦 Step 5: Enable live motion only after the dry run is clean

When the dry run is stable, launch with motion enabled.

```bash
./run_recorder.sh --home
```

Or for direct mirroring:

```bash
./run_recorder.sh --no-home
```

The GUI requires motion confirmation first. That is intentional. Treat it as a final safety gate, not an inconvenience.

## ▶️ Step 6: Run one real teleop episode

Recommended first live episode:

1. Confirm the follower is clear to move.
2. Open the dashboard.
3. Confirm motion from the UI.
4. Start recording by DI0 or the Start button.
5. Move the leader through a short, obvious trajectory.
6. Stop the episode.
7. Verify that the recorder saved the session correctly.

The first live episode is still a validation run. Keep it short.

## 📁 Step 7: Inspect the resulting LeRobot dataset

The current stack records directly to LeRobot format. Inspect:

| Path | Expected result |
| --- | --- |
| `recordings_lerobot/session_<timestamp>/meta/info.json` | Dataset summary |
| `data/chunk-000/episode_*.parquet` | Tabular state and action frames |
| `videos/chunk-000/.../episode_*.mp4` | Encoded camera video |

If you only see metadata and no episode payloads, treat the run as failed and fix that before another teleop attempt.

## ⚙️ Step 8: Tune control parameters only after you have a baseline

The launcher exposes the key motion parameters:

| Flag | Use it to change |
| --- | --- |
| `--max-velocity` | ServoJ speed ceiling |
| `--max-accel` | ServoJ acceleration ceiling |
| `--alignment-speed` | Home/catch-up motion speed |
| `--camera-fps` | Camera throughput and recording cost |

Do not tune these blindly. Change one parameter at a time, then repeat the same short validation episode.

## 🔁 Step 9: Use the workflow for repeated teleop collection

After the first validation run succeeds, the repeatable operator workflow is:

1. Launch the stack.
2. Confirm motion.
3. Start an episode.
4. Teleoperate the action.
5. Stop the episode.
6. Repeat.
7. Shut down with Ctrl+C so the stack flushes cleanly.

That loop is the intended operating mode.

## 🔍 Troubleshooting

### Follower movement is jerky

Return to dry-run mode first. Then lower throughput pressure by reducing camera FPS or switching to simulated cameras while you isolate whether the problem is control or image load.

### Motion confirmation never clears

The destination writer is probably not publishing a usable state transition yet.

### Episodes save but camera panels are blank

Use the replay or synthetic camera path and isolate the RealSense problem separately.

### The wrong robot moves

Check the actual IPs. A placeholder like `192.168.1.x` left in your local command or settings is enough to invalidate the whole rig mapping.

## 🔗 Related documentation

- [leader_follower README](../../../data-pipeline/capture/ur/leader_follower/README.md)
- [Recording with the vendor-agnostic Episode Recorder](using-episode-recorder.md)
- [Running VLA training and evaluation](../training/using-vla-training-and-evaluation.md)
