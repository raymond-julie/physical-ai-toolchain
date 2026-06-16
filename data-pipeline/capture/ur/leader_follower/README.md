# UR Leader/Follower Teleop + Record

Mirror a manually operated source UR5e ("leader") onto a programmatically
controlled destination UR5e ("follower") while recording each teleoperation
episode as a [LeRobotDataset](https://github.com/huggingface/lerobot) (parquet
state/action rows + encoded mp4 videos).

A single launcher starts the source reader, destination writer, LeRobot
recorder, camera drivers, and a Flask web dashboard. A rising edge on tool
digital input 0 (DI0) — or the dashboard START button — toggles full-speed
mirroring and recording on; a second press toggles it off.

> [!WARNING]
> This tool commands physical robot motion. Keep the workspace clear, keep an
> e-stop within reach, and confirm motion from the dashboard before the
> follower moves. Motion is disabled by default (`--no-motion`).

## 📋 System Context

| Node | Example IP | Role |
| --- | --- | --- |
| Source UR5e | `192.168.1.80` | Leader robot (manually operated) |
| Destination UR5e | `192.168.1.90` | Follower robot (programmatically controlled) |
| Workstation | `192.168.1.71` | Ubuntu host running this stack |
| Monitor PC | any LAN host | Optional monitoring via the web dashboard |

> [!NOTE]
> The `192.168.1.x` addresses are documented example defaults. Override them
> with `--source-ip` / `--dest-ip` or the dashboard settings.

## 🚀 Quick Start

```bash
# Install ROS 2 + Python dependencies (apt-based hosts).
./install_dependencies.sh

# Source ROS 2, then launch the full stack (direct mirroring, motion enabled).
./run_recorder.sh

# Dry run with no robot motion.
./run_recorder.sh --no-motion

# Align to a home pose at start/end instead of mirroring directly.
./run_recorder.sh --home

# List every launch option.
./run_recorder.sh --help
```

The dashboard is served on `http://<host-ip>:8080` (the launcher prints the
detected address at startup).

## 🧩 Components

| File | Role |
| --- | --- |
| `source_reader.py` | Reads leader joints (RTDE), gripper (Robotiq socket), and tool DI0/DI1; publishes `/mirror/*`. |
| `destination_writer.py` | State machine (align → idle → mirror → return) driving the follower via `servoJ`; publishes `/recorder/active`. |
| `lerobot_recorder_node.py` | Records each DI0 cycle as one LeRobotDataset episode (parquet + mp4). |
| `recorder_node.py` | Legacy ROS 2 bag recorder with timestamp-continuity verification (retained for raw capture). |
| `ros_bag_recorder.py` | Thin `ros2 bag record` subprocess wrapper used by `recorder_node.py`. |
| `gui_node.py` | Flask + SocketIO dashboard: live state, MJPEG camera previews, recording list, settings. |
| `video_to_camera.py` | Replays an mp4 (or a synthetic pattern) on the camera topics for no-hardware testing. |
| `local_retention.py` | Timer node that prunes aged on-disk episode chunks while keeping metadata. |

## 🔌 ROS 2 Topics

| Topic | Type | Publisher |
| --- | --- | --- |
| `/mirror/joint_states` | `sensor_msgs/JointState` | Source reader |
| `/mirror/gripper/position` | `std_msgs/Float64` | Source reader |
| `/mirror/gripper/is_closed` | `std_msgs/Bool` | Source reader |
| `/mirror/tool_digital_input_0` | `std_msgs/Bool` | Source reader |
| `/mirror/tool_digital_input_1` | `std_msgs/Bool` | Source reader |
| `/joint_states` | `sensor_msgs/JointState` | Destination writer |
| `/recorder/active` | `std_msgs/Bool` | Destination writer |
| `/destination/state` | `std_msgs/String` | Destination writer |

## ⚙️ Key Parameters

| Parameter | Default | Unit | Description |
| --- | --- | --- | --- |
| `robot_ip` (source) | `192.168.1.80` | — | Leader RTDE address |
| `robot_ip` (dest) | `192.168.1.90` | — | Follower RTDE address |
| `use_home` | `true` | bool | Align to home at start/end; `false` mirrors directly |
| `alignment_speed` | `0.1` | rad/s | Joint velocity during alignment and catch-up |
| `alignment_threshold` | `0.02` | rad | Per-joint error to declare alignment complete |
| `max_velocity` | `1.5` | rad/s | `servoJ` velocity during mirroring |
| `max_acceleration` | `3.0` | rad/s² | `servoJ` acceleration during mirroring |
| `servo_time` | `0.008` | s | `servoJ` control period (125 Hz) |
| `stale_timeout` | `0.2` | s | Hold the servo target if no fresh mirror sample arrives |
| `fps` | `30` | Hz | LeRobot episode sampling rate |
| `min_episode_frames` | `5` | frames | Episodes shorter than this are discarded as stray toggles |

## 🎥 Running Without RealSense Hardware

When the lab cameras are unavailable (or for CI / blob-sync smoke tests), the
launcher replaces `realsense2_camera` with `video_to_camera.py`, which
republishes an mp4 (or a synthetic pattern) on the same camera topics the
recorder subscribes to:

```bash
# Replay a previously recorded LeRobot mp4 chunk as camera1.
./run_recorder.sh --no-motion \
  --video ./recordings_lerobot/local/ur5_mirror/videos/chunk-000/observation.images.color/episode_000000.mp4

# Moving-bars synthetic pattern when no mp4 is available.
./run_recorder.sh --no-motion --synthetic-camera

# Dual-camera replay.
./run_recorder.sh --no-motion --video cam1.mp4 --video2 cam2.mp4
```

Combine with `LEROBOT_ROOT=/cloud-sync/lerobot-recordings` to round-trip frames
through an ACSA-backed PVC and verify they reach the `lerobot-recordings` blob
container.

## 🔄 Recording Lifecycle

1. The launcher starts all nodes and the dashboard, then waits for motion confirmation.
2. After confirmation, the follower optionally aligns to the home pose (`--home`).
3. A DI0 rising edge (or the START button) begins catch-up, then full-speed mirroring, and starts a LeRobot episode.
4. A second DI0 press stops mirroring and calls `save_episode()` (parquet + mp4).
5. With `--home`, the follower returns to home; otherwise it holds position. The cycle repeats from step 3.

## 📦 Dependencies

| Dependency | Notes |
| --- | --- |
| ROS 2 Humble / Jazzy | Ubuntu 22.04 / 24.04 or Jetson |
| Python | >= 3.10 |
| `ur_rtde` | Leader/follower RTDE control (`pip install ur_rtde`) |
| `lerobot` | Dataset writer (`pip install lerobot`) |
| `cv_bridge` | ROS image → numpy conversion (`ros-${ROS_DISTRO}-cv-bridge`) |
| `opencv-python`, `numpy`, `pandas` | Image handling and dataset writer support |
| `flask`, `flask-socketio` | Web dashboard |
| `rclpy`, `sensor_msgs`, `std_msgs` | Provided by the system ROS 2 install |

> [!NOTE]
> On ROS 2 Humble, `cv_bridge` is built against NumPy 1.x, so `numpy<2` is
> required; on Jazzy it is built against NumPy 2.x. `install_dependencies.sh`
> applies the correct constraint based on `$ROS_DISTRO`.

## 🧪 Tests

Behavior tests cover the pure logic (joint interpolation and alignment math,
episode framing, retention pruning) with the ROS 2 modules stubbed, so no
hardware or ROS install is required:

```bash
cd data-pipeline/capture/ur/leader_follower
python -m pytest tests -q
```
