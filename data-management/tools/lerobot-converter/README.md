# MCAP to LeRobot Converter

Convert ROS 2 MCAP teleop recordings into LeRobot v2.1 datasets, with companion
inspection CLIs for verifying a recording before conversion.

## 📋 Prerequisites

- Python 3.10 or newer
- Install dependencies into a standalone virtualenv:

```bash
python -m pip install -r requirements.txt
```

> [!NOTE]
> Video encoding uses PyAV (`libx264`); no system ffmpeg install is required.
> The decoder package `mcap-ros2-support` is imported as `mcap_ros2`.

## 🚀 Quick Start

Convert every `<timestamp>/<timestamp>.mcap` episode under a recordings directory
into a LeRobot v2.1 dataset:

```bash
python convert_mcap_to_lerobot.py \
    --src /home/aloha/recordings \
    --out /data/pick-place-v1 \
    --repo-id pick-place-v1 \
    --task "pick up the block and place it in the bin" \
    --fps 15 \
    --overwrite
```

The converter samples every topic onto a uniform timeline at `--fps` using the
MCAP `log_time` clock with nearest-neighbor selection, then writes `meta/`,
`data/chunk-000/`, and per-camera MP4 video under `videos/chunk-000/`.

## 📦 Tools

| Tool                        | Purpose                                                              |
|-----------------------------|---------------------------------------------------------------------|
| `convert_mcap_to_lerobot.py`| Convert MCAP episodes to a LeRobot v2.1 dataset (MP4 video)          |
| `inspect_mcap.py`           | List channels, schemas, encodings, statistics, and per-topic counts |
| `inspect_messages.py`       | Decode the first message on each requested topic                    |
| `inspect_wrench_gripper.py` | Summarize TCP wrench, gripper joint range, and tool-grabbed states   |
| `check_motion.py`           | Report per-joint motion range of the left vs right follower arms     |

Each inspection CLI takes the MCAP path as a positional argument:

```bash
python inspect_mcap.py /home/aloha/recordings/20260603_141502/20260603_141502.mcap
```

## 🎛️ Dataset Layout

The converter targets a bimanual UR embodiment: two 7-DoF follower arms (6 joints
plus a gripper) and four Orbbec cameras.

| LeRobot feature                       | Source topic                                              | Shape |
|---------------------------------------|----------------------------------------------------------|-------|
| `observation.state`                   | `*/follower/arm/joint_states` (left + right)             | [14]  |
| `action`                              | `*/follower/action/MoveToPosition` (left + right)        | [14]  |
| `observation.tcp_wrench.{left,right}` | `*/follower/arm/tcp_wrench`                              | [6]   |
| `observation.images.cam_*`            | `/cam_*/camera/left_image/distorted/compressed`          | video |

Joints per arm: `shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`, `wrist_2`,
`wrist_3`, `gripper`.

## ⚙️ Configuration

| Argument        | Default                 | Description                                       |
|-----------------|-------------------------|---------------------------------------------------|
| `--src`         | `/home/aloha/recordings`| Directory of `<timestamp>/<timestamp>.mcap` dirs  |
| `--out`         | required                | Output dataset root                               |
| `--repo-id`     | required                | LeRobot repository id recorded in `info.json`     |
| `--task`        | required                | Task description stored in `tasks.jsonl`          |
| `--fps`         | `30`                    | Output frame rate (training datasets used 15)     |
| `--date-prefix` | none                    | Only convert episode dirs starting with a prefix  |
| `--limit`       | none                    | Convert at most N episodes                         |
| `--overwrite`   | off                     | Replace an existing output dataset                |

## 🧪 Tests

Pure-logic tests (timeline resampling and feature assembly) live in
[../tests](../tests) and run without MCAP files or ffmpeg:

```bash
uv run pytest data-management/tools/tests -v
```

## 🔗 Related

- [../README.md](../README.md) — data management tools index
- [../../specifications/dataset-curation.specification.md](../../specifications/dataset-curation.specification.md) — Convert/Merge contracts
