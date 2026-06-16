# UrDualRecorder

Dual-arm Universal Robots **recorder**. It reads the live state of two follower
UR arms over RTDE — joint positions plus each follower's Robotiq 2F-85 gripper
state — and records synchronized episodes (arm states + every camera) as a
LeRobotDataset. No teleoperation, no mirroring, no motion commands: the recorder
only **observes**.

This is a standalone (no-ROS) sibling of [`../leader_follower/`](../leader_follower/README.md)
and [`../episode_recorder/`](../episode_recorder/README.md), built around the
TrainMyBot device topology in `/etc/trainmybot/config_v3.yaml`.

## What it does

- Reads each follower arm's joint positions over RTDE (read-only; never takes
  the control lock).
- Reads each follower's Robotiq 2F-85 gripper position (0 = open, 255 = closed)
  and derives an open/closed flag from `gripper.closed_threshold`.
- Captures every configured Orbbec camera.
- Records `observation.state` (joints + gripper position per arm), per-arm
  gripper-closed flags, and each camera stream as a LeRobotDataset
  (parquet + mp4).
- Serves a web dashboard for toggling recording and viewing live previews and
  arm/gripper status.

## Topology (from `config_v3.yaml`)

Only the **follower** arms are recorded:

| Side  | Follower (read-only)         |
|-------|------------------------------|
| left  | `arm_left_follower` .11      |
| right | `arm_right_follower` .13     |

Cameras (`cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`) are opened
by serial via the Orbbec SDK (synthetic moving-bar fallback when the SDK is
absent).

## Layout

```text
dual_recorder/
├── ur_dual_recorder/
│   ├── config.py          # load config_v3.yaml + app overlay; build pairs
│   ├── arm_reader.py      # read-only follower state (joints + gripper + DI0)
│   ├── robotiq.py         # Robotiq 2F-85 socket client (read)
│   ├── cameras.py         # Orbbec capture (synthetic fallback)
│   ├── recorder.py        # multi-arm LeRobotDataset writer (EpisodeRecorder)
│   ├── app.py             # orchestrator: arm readers + cameras + recorder + GUI
│   ├── __main__.py        # CLI entrypoint
│   └── web/               # Flask dashboard (server + templates + static)
├── config/app.yaml        # app settings overlay (gripper read, recording)
├── run_dual_recorder.sh   # launcher
├── requirements.txt
└── setup.py
```

> [!NOTE]
> The shelved teleop modules (`analog.py`, `ur_interface.py`, `teleop.py`)
> remain in the tree for when mirroring is re-enabled, but are not imported in
> recording-only mode.

## Gripper state

Each follower's Robotiq 2F-85 is read over its socket port (`63352`). The raw
position (0..255) is normalized to a 0..1 fraction for `observation.state`, and
`gripper.closed_threshold` (default `128`) decides the boolean closed flag
recorded as `observation.<arm>.gripper_is_closed`.

## Install

```bash
cd data-pipeline/capture/ur/dual_recorder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install the Orbbec SDK wheel for your platform from
# https://github.com/orbbec/pyorbbecsdk
```

## Run

```bash
# Web GUI + recording, DI0 trigger enabled:
./run_dual_recorder.sh

# No web dashboard:
./run_dual_recorder.sh --no-web

# Disable the DI0 record trigger (GUI Record button only):
./run_dual_recorder.sh --no-di0-trigger

# Don't record, just preview state/cameras:
./run_dual_recorder.sh --no-record

# Point at a different config:
./run_dual_recorder.sh --config /path/to/config_v3.yaml
```

Open `http://<host>:8080/` for the dashboard. Recording toggles on a follower
**DI0** press or the GUI **Record** button.

## Recorded schema

For the two-arm config, per frame:

| Key                                       | dtype   | shape     | meaning                              |
|-------------------------------------------|---------|-----------|--------------------------------------|
| `observation.state`                       | float32 | (14,)     | per arm: 6 joints + gripper position |
| `observation.<arm>.gripper_is_closed`     | bool    | (1,)      | one per follower arm                 |
| `observation.images.<camera_id>`          | video   | (H, W, 3) | one per configured camera            |

Episodes land in `recordings_lerobot/session_<timestamp>/`.

## GMSL camera driver (Jetson MIC-733AO)

The `gmsl-driver-mic-733ao/` directory holds the host-side GMSL camera watchdog
(`gmsl_watchdog.py`) and reconnect helpers used on the Jetson carrier board. The
compiled kernel modules (`*.ko`, `*.dtbo`) are platform-specific build artifacts
and are **not** committed; build them on the target host. See
[`gmsl-driver-mic-733ao/README.md`](gmsl-driver-mic-733ao/README.md).

## Notes

- Read-only: the recorder uses `RTDEReceiveInterface` only, so it can run
  alongside another program controlling the arms.
- All hardware libraries (`ur_rtde`, `pyorbbecsdk`, `lerobot`) are imported
  defensively; on a dev box without them the app still starts with synthetic
  cameras, `DOWN` arm/gripper status, and recording disabled.
