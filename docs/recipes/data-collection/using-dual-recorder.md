---
title: Recording Dual-Arm UR Data with the Dual Recorder
description: End-to-end tutorial for running the standalone dual-arm UR recorder, validating the dashboard, and producing LeRobot episodes from two follower arms and four cameras
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: tutorial
keywords:
  - dual recorder
  - lerobot
  - ur5e
  - orbbec
  - data collection
estimated_reading_time: 22
---

This tutorial shows how to use `dual_recorder`, the standalone dual-arm capture app that reads two follower UR arms and multiple cameras without commanding motion. Use it when the goal is high-quality data capture, not teleoperation or replay.

## 📋 Prerequisites

| Requirement | Details |
| --- | --- |
| Robots | Two reachable follower arms with RTDE read access |
| Grippers | Robotiq 2F-85 grippers on the followers if gripper state is required |
| Cameras | Orbbec cameras wired and visible to the host, or synthetic fallback accepted for dry validation |
| Host config | `/etc/trainmybot/config_v3.yaml` populated for the rig |
| Python | Python 3.10+ |
| Network plan | Replace every `192.168.1.x` placeholder with the actual robot or workstation IPs before running |

## 🧱 Step 1: Verify the rig configuration before starting the app

The dual recorder assumes the device topology is already described in `config_v3.yaml`. Validate that file first.

Confirm these fields exist:

| Area | What to verify |
| --- | --- |
| Follower arms | `arm_left_follower` and `arm_right_follower` entries with real robot IPs |
| Camera devices | `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` camera definitions |
| Serials | Camera serials match what the Orbbec SDK reports on the host |

If the config is wrong, the recorder may still start, but you will get missing arms, wrong cameras, or synthetic fallbacks instead of real data.

## 📦 Step 2: Install dependencies

```bash
cd data-pipeline/capture/ur/dual_recorder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the host does not already have the Orbbec wheel, install the correct `pyorbbecsdk` build for the platform. If you skip that, the UI works but the feeds are synthetic.

## 🔌 Step 3: Prove basic arm and camera visibility without recording

Start the app in preview mode first.

```bash
./run_dual_recorder.sh --no-record
```

Open the dashboard at `http://<host>:8080`.

Validate:

| Signal | What you should see |
| --- | --- |
| Arm status | Both followers marked connected |
| Joint values | Live updates when you move the arms manually |
| Gripper state | Position changes and closed/open flag updates |
| Camera panels | Four live feeds or clearly marked synthetic feeds |

Do not record until this preview mode is healthy. Recording only makes sense after the inputs are stable.

## ▶️ Step 4: Start a normal recording session

Run the recorder normally.

```bash
./run_dual_recorder.sh
```

You now have two supported trigger paths:

| Trigger | Behavior |
| --- | --- |
| Follower DI0 | Toggles episode recording on the configured tool input |
| GUI Record button | Toggles the same state from the dashboard |

Recommended first capture:

1. Start with a short 10-15 second episode.
2. Move both arms through clearly visible motion.
3. Open and close both grippers during the take.
4. Stop the episode.

The first test should be short because you are validating the pipeline, not collecting production data yet.

## 📁 Step 5: Inspect what was written to disk

Episodes are written under `recordings_lerobot/session_<timestamp>/`.

Check the structure:

```text
recordings_lerobot/
└── session_<timestamp>/
    ├── data/
    ├── meta/
    └── videos/
```

What to inspect:

| Path | Expected result |
| --- | --- |
| `meta/info.json` | Dataset summary with fps, total episodes, and feature schema |
| `data/chunk-000/episode_*.parquet` | Per-episode tabular state records |
| `videos/chunk-000/.../episode_*.mp4` | Encoded video clips for each camera |

If the directory contains only metadata and no episode parquet or mp4 files, the trigger fired but a real episode was not captured successfully.

## 🧪 Step 6: Run through the common operating modes

Use these modes once each so the team knows what they do before a real collection session.

### GUI only, no DI0 trigger

```bash
./run_dual_recorder.sh --no-di0-trigger
```

Use this when the physical tool input is unavailable or noisy.

### Headless, no web dashboard

```bash
./run_dual_recorder.sh --no-web
```

Use this for scripted or service-managed runs where the app is supervised externally.

### Alternate config file

```bash
./run_dual_recorder.sh --config /path/to/config_v3.yaml
```

Use this when one workstation manages multiple robot cells.

## 🔄 Step 7: Connect blob_sync after local recording works

The dual recorder is the producer. `blob_sync` is the uploader. Do not debug both at once.

Once local recording is healthy, connect the upload flow:

1. Keep recording into `recordings_lerobot`.
2. Configure `blob_sync` with the container SAS URL through `config.yaml` or `BLOB_SYNC_CONTAINER_URL`.
3. Run `blob_sync` in `--check` mode first.
4. Run one completed session through `--once`.

That sequence proves the recorder and uploader independently before you depend on them together.

## ✅ Step 8: Define the operator workflow for real collection

A stable collection session usually looks like this:

1. Launch preview mode and verify all inputs.
2. Restart in normal mode.
3. Record a short calibration episode.
4. Inspect the output folder.
5. Record the real session in repeated episodes.
6. Run upload or post-processing only after local capture is finished.

That order catches hardware issues early and keeps partially broken sessions out of the dataset.

## 🔍 Troubleshooting

### One arm is down

Check RTDE connectivity and verify the actual robot IPs replaced the `192.168.1.x` placeholders in your local config and launch arguments.

### All cameras are synthetic

The app is falling back because the Orbbec SDK is unavailable or the devices could not be opened.

### Dashboard works but no episode files appear

Verify the trigger path, then record a longer test. Very short episodes can be discarded by the minimum frame threshold or may never accumulate enough buffered frames to save.

### Gripper flags never change

Verify the Robotiq socket connection and confirm the configured port is correct.

## 🔗 Related documentation

- [UrDualRecorder README](../../../data-pipeline/capture/ur/dual_recorder/README.md)
- [Using Camera Streamer and the URCap Pendant Viewer](using-camera-streamer-and-urcap.md)
- [Recording with the vendor-agnostic Episode Recorder](using-episode-recorder.md)
