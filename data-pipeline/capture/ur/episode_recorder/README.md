# episode_recorder

Vendor-agnostic, multi-robot LeRobot dataset recorder. Each episode — started and stopped by a physical button **or** a web GUI button — is written as a LeRobotDataset (parquet + mp4 video) capturing the joint and gripper state of every configured robot plus color (and optionally depth) frames from every configured camera. No robot is ever commanded; both arms are observed read-only.

The robot vendor, gripper vendor, robot count, joint count, and camera count are all configurable. New hardware is added by writing a small driver module and registering it under a name — the recorder and launcher are untouched.

## 📋 Prerequisites

- ROS 2 (Humble on Ubuntu 22.04, Jazzy on Ubuntu 24.04)
- Python 3.10+
- Python dependencies from [requirements.txt](requirements.txt): `ur_rtde`, `nats-py`, `flask`, `numpy`, `pandas`, `opencv-python`, `lerobot`, `aiortc`, `av`
- ROS packages from the system install: `rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`, and (for local cameras) `realsense2_camera`

Install everything on an apt-based host with [install_dependencies.sh](install_dependencies.sh):

```bash
cd data-pipeline/capture/ur/episode_recorder
./install_dependencies.sh
```

## 🚀 Quick Start

```bash
cd data-pipeline/capture/ur/episode_recorder
source /opt/ros/${ROS_DISTRO}/setup.bash

# Default: 2x UR5e + Robotiq, 2x RealSense, RGB only, 15 fps, GUI on :8080
./run_episode_recorder.sh

# Robot 2 has no gripper
./run_episode_recorder.sh --gripper2-driver none

# Record depth too
./run_episode_recorder.sh --depth --camera-fps 30

# Headless setup with no physical button — GUI is the only trigger
./run_episode_recorder.sh --no-tool-trigger
```

Press the physical DI0 on Robot1 or click **Record** at `http://<this-host>:8080/` to start and stop episodes. Stop the stack with **Ctrl+C** — the launcher tears down all subprocesses and the recorder flushes any in-progress episode before exit. Run `./run_episode_recorder.sh --help` for every option.

## 🗂️ Layout

```text
episode_recorder/                 # importable Python package
├── drivers/                      # vendor-specific code lives here
│   ├── base.py                   # ABCs: RobotStateDriver / GripperDriver
│   ├── registry.py               # name -> class lookup
│   ├── ur_rtde.py                # Universal Robots via ur_rtde (direct RTDE)
│   ├── nova.py                   # Wandelbots Nova v2 (NATS subscriber)
│   ├── robotiq.py                # Robotiq 2F socket protocol
│   └── noop.py                   # "no gripper" stub
├── nodes/                        # ROS 2 nodes (vendor-agnostic)
│   ├── robot_reader.py           # generic robot+gripper reader
│   ├── episode_recorder.py       # N-robot, N-camera dataset writer
│   ├── trigger_tool_io.py        # physical-DI button trigger
│   ├── trigger_gui.py            # web Record button (fallback)
│   └── nova_camera_bridge.py     # Nova WebRTC -> ROS Image bridge
└── web/                          # GUI assets (templates + static)
config/robots.example.yaml        # sample multi-robot config
nova.example.env                  # Nova credentials/connection template
run_episode_recorder.sh           # one-script launcher
install_dependencies.sh           # apt + pip + ROS dependency installer
keep_nova_session_alive.py        # holds a Nova motion-group session open
requirements.txt
setup.py                          # optional: `pip install -e .`
```

This is a portable, driver-pluggable evolution of the original flat `leader_follower` UR teleop recorder: the robot vendor is no longer hardcoded, the robot/joint/camera counts are parameters, and the recording trigger is a separate, swappable node.

## 🏗️ Architecture

```text
                ┌──────────────────────┐       ┌────────────────────┐
   physical ───►│ trigger_tool_io node │──┐    │   trigger_gui node │◄── web Record button
   DI0 button   └──────────────────────┘  │    └────────────────────┘
                                          ▼    ▼
                                  /recorder/active (std_msgs/Bool)
                                          │
        ┌───────────┐                     ▼                  ┌────────────────────┐
        │robot_reader│──/robot1/*  ─► ┌──────────────────┐ ◄─│ realsense2_camera  │
        │  (driver=  │                │ episode_recorder │   │ (camera1, camera2) │
        │  ur_rtde)  │                │       node       │   └────────────────────┘
        └───────────┘                 └──────────────────┘
        ┌───────────┐                          │
        │robot_reader│──/robot2/*  ─►          ▼
        │  (driver=  │                 LeRobotDataset
        │  ur_rtde)  │                 (parquet + mp4)
        └───────────┘
```

Either trigger source can drive recording. Both stay in sync because each subscribes to `/recorder/active` and publishes `!current` on its own event, so multiple trigger sources coexist without state divergence.

## 🔌 Adding a Robot Vendor

1. Create `episode_recorder/drivers/<vendor>.py`.
2. Implement `RobotStateDriver` (and optionally `GripperDriver`) from [episode_recorder/drivers/base.py](episode_recorder/drivers/base.py).
3. Call `register_state_driver("<vendor>", YourDriver)` at module bottom.
4. Add `from . import <vendor>` to the `_autoload` block in [episode_recorder/drivers/registry.py](episode_recorder/drivers/registry.py).
5. Launch with `--robot1-driver <vendor>`.

No changes to the recorder node or to the launcher (beyond the CLI flag) are required. Joint count is configurable via `joints_per_robot`; the recorder generalises to N robots and N cameras automatically.

## 🎬 Recording Controls

| Source | Behaviour |
| --- | --- |
| Physical DI0 | `trigger_tool_io` listens on `/robot1/digital_input/di0` (configurable). A rising edge toggles `/recorder/active`. |
| Web GUI | `trigger_gui` serves a Record/Stop button plus live MJPEG previews of every configured camera at `http://<host>:8080/`. Same toggle semantics as DI0. |
| Direct topic | Any external publisher can drive `/recorder/active` (`std_msgs/Bool`). |

## 🛰️ State Source — RTDE vs Nova

The robot reader can fetch joint state from either the UR controller directly (RTDE) or from a Wandelbots Nova deployment (NATS). Pick with `--state-source rtde|nova` — no second recorder is required.

| Aspect | `--state-source rtde` (default) | `--state-source nova` |
| --- | --- | --- |
| Driver | `ur_rtde` | `nova` |
| Connection | TCP/RTDE to each UR controller | NATS subscription `nova.v2.cells.{cell}.controllers.{ctrl}.state` |
| Joint positions | From `getTargetQ` / `getActualQ` | From `motion_groups[0].joint_position` |
| Joint velocities | From `getTargetQd` / `getActualQd` | Not exposed by Nova v2 state — recorded as zeros |
| Tool DI0 button | Via `getActualDigitalInputBits` | Not in Nova v2 state — physical-button trigger auto-disabled |
| Robotiq gripper | Direct socket `:63352` | Direct socket `:63352` (Nova does not proxy the gripper port) |
| Co-existence with Nova | Nova owns the RTDE lock | Safe — Nova does the talking |

Nova-mode example (lab defaults are baked into the launcher, so the minimal command is just):

```bash
./run_episode_recorder.sh --state-source nova
```

To override individual settings on the command line:

```bash
./run_episode_recorder.sh \
    --state-source nova \
    --nova-nats-url nats://192.168.1.244:31422 \
    --nova-cell cell \
    --nova-ctrl1 ur5-left  --robot1-name robot1 --robot1-ip 192.168.1.80 \
    --nova-ctrl2 ur5-right --robot2-name robot2 --robot2-ip 192.168.1.90
```

Secret values (NATS user/password or a JWT credentials file) live in a local `nova.env` next to the launcher; [nova.example.env](nova.example.env) ships as a template. `nova.env` is gitignored (rule: `nova.env` plus the repo-wide `*.env`); only the `*.example.env` template is tracked. Value precedence is **defaults → `nova.env` → CLI flags**, so anything in the file can be overridden ad-hoc with `--nova-*` arguments. Point the launcher at a different file with `--nova-env-file <path>`.

> [!NOTE]
> The discovery procedure for the NATS endpoint and credentials on a fresh Nova
> deployment is documented in the lab reference
> `schaeffler_robotics_workloads/apps/ur_mqtt_wl/README.md`, section 5.

> [!WARNING]
> Nova v2's controller-state stream omits the UR tool digital-input bits, so the
> launcher auto-applies `--no-tool-trigger` in Nova mode. The Web GUI Record
> button remains the trigger.

## 📷 Camera Source — Local USB vs Nova WebRTC

Color frames come from either the host's USB-attached RealSense cameras (default, via `realsense2_camera`) or from Wandelbots Nova's RealSense camera-manager app (`http://<nova-host>/cell/realsense`) over **WebRTC**. Select with `--camera-source local|nova`. When `--state-source nova` is selected the launcher auto-picks `--camera-source nova` so the whole pipeline runs against Nova without contending for USB devices; pass `--camera-source local` to override.

The Nova bridge node (`episode_recorder.nodes.nova_camera_bridge`):

1. discovers cameras via `GET /api/devices/` (or uses the `--nova-cam-serials` list / `NOVA_CAM_SERIALS` env var);
2. performs the server-offer WebRTC handshake against `POST /api/webrtc/{offer,answer}`;
3. decodes each color track with PyAV and republishes it as `sensor_msgs/Image` on `/cameraN/cameraN/color/image_raw` — the exact topic name `realsense2_camera` would use — so the recorder and Web GUI consume Nova frames transparently.

| Setting | Default |
| --- | --- |
| `NOVA_CAM_API_BASE` / `--nova-cam-api-base` | `http://192.168.1.71/cell/realsense` |
| `NOVA_CAM_SERIALS` / `--nova-cam-serials` | auto-discover via `/api/devices/` |
| `NOVA_CAM_STREAM_TYPES` / `--nova-cam-stream-types` | `color` |

> [!NOTE]
> The bridge currently publishes only the color track; `--depth` is ignored under
> `--camera-source nova`.

## 🧱 Dataset Frame Schema

For `robot_namespaces=[robot1, robot2]`, `joints_per_robot=6`, two cameras, and no depth:

| Key | dtype | shape |
| --- | --- | --- |
| `observation.state` | float32 | (14,) |
| `action` | float32 | (14,) (= state) |
| `observation.robot1.gripper_is_closed` | bool | (1,) |
| `observation.robot2.gripper_is_closed` | bool | (1,) |
| `observation.images.color_0` | video/png | (H, W, 3) |
| `observation.images.color_1` | video/png | (H, W, 3) |

The state vector layout per robot is `[joints[0..J-1], gripper_position]`. `action` is a copy of `observation.state` (no commanded action is available — the recorder is read-only), provided because LeRobotDataset conventionally expects the key.

## 🧩 Design Notes

| Aspect | `leader_follower` (original) | `episode_recorder` |
| --- | --- | --- |
| Robot vendor | Hardcoded UR + Robotiq | Driver-pluggable |
| Number of robots | 2 (source/destination) | N (`robot_namespaces`) |
| Joint count | Hardcoded 6 | Parameter `joints_per_robot` |
| Number of cameras | Hardcoded 2 | N (`image_topics`) |
| Recording trigger | Coupled into source reader | Separate node, swappable |
| GUI fallback button | No | Yes (`trigger_gui`) |
| Package layout | Flat scripts | Importable Python package |
| Robot motion | Source → destination mirror | None (read-only) |

## 🧪 Tests

Behavior tests for the driver registry and the read-only observation-assembly logic live in [tests/](tests). They stub the optional native/SDK modules (`rclpy`, `ur_rtde`, `nats`, `aiortc`, `av`, `cv2`) so the suite needs no ROS install, robot hardware, or network:

```bash
cd data-pipeline/capture/ur/episode_recorder
python -m pytest tests -q
```
