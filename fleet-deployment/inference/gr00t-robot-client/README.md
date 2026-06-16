# GR00T Dual-Arm UR5e Inference Client

Closes the control loop that the [GR00T Control UI](../gr00t-control-ui) does
not. The UI only **starts/stops** the policy server (it scales the Kubernetes
Deployment); it never drives the robots. This client is the missing piece: it
reads the cameras and arm joints, queries the GR00T policy, and streams the
predicted joint targets to the two UR5e follower arms.

```text
cameras (ur-camera-streamer :8000) ─┐
                                     ├─► observation ─► GR00T policy :5555
arms (RTDE getActualQ) ─────────────┘                        │
                                                             ▼
UR5e arms ◄──────────── servoJ(joint targets) ◄──── action.robotN_arm (16×6)
```

> [!WARNING]
> The GitOps `tegra` HelmRelease deploys this client with `robotClient.execute: true`
> and `robotClient.assumeYes: true`, which means **the arms move automatically** once
> Flux reconciles the cluster — there is no interactive confirmation in-cluster.
> Keep an e-stop within reach and the workspace clear before enabling it.

## 🤖 What it does

1. **Observe** — grabs a JPEG snapshot from each of the four Orbbec cameras
   (`color_0..3`) and reads the 6 joint angles of each follower arm over RTDE.
2. **Infer** — sends the observation to the GR00T N1.5 policy server over ZMQ
   and receives a 16-step action chunk per arm.
3. **Act** — streams the predicted joint targets to both arms with `servoJ`,
   using a receding horizon (execute the first few steps, then re-query).

The GR00T ZMQ wire format (msgpack + numpy `.npy`) is reimplemented locally, so
the robot host only needs `pyzmq`, `msgpack`, `numpy`, `requests`, OpenCV and
`ur_rtde` — **not** the heavy `isaac-gr00t` package.

## 🛡️ Safety model

Driving real arms, so the defaults are conservative:

| Guard                | Behavior                                                                                                                                                       |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dry run by default   | Without `--execute` it queries and logs only; arms never move.                                                                                                  |
| Execute confirmation | `--execute` prompts for a typed `MOVE` confirmation (skipped only with `--assume-yes`).                                                                         |
| First-pose gate      | Refuses to start if the first predicted pose is more than `--start-threshold` rad (default 0.30) from the measured joints. Override with `--allow-jump`.        |
| Per-step clamp       | Each command is at most `--max-joint-step` rad (default 0.03) from the previous one, capping joint speed.                                                        |
| Absolute clamp       | With `--metadata metadata.json`, targets are clamped to the trained action range.                                                                               |
| Clean shutdown       | `servoStop` + `stopScript` + `disconnect` run on every exit (incl. Ctrl-C).                                                                                     |

## 📋 Prerequisites

```bash
pip install -r requirements.txt
```

The policy service is a ClusterIP. From a host outside the cluster, forward it:

```bash
kubectl port-forward -n default svc/gr00t-gr00t-inference 5555:5555
```

The arms must be in **remote control** mode and reachable, and the
`ur-camera-streamer` must be running (it is `hostNetwork` on the GR00T node,
port `8000`).

## 🚀 Usage

Confirm connectivity (no movement):

```bash
python3 robot_inference_client.py --ping
python3 robot_inference_client.py --once \
    --camera-url http://192.168.1.10:8000 \
    --task "pick up the red block and place it in the box"
```

Dry run the full loop (still no movement):

```bash
python3 robot_inference_client.py \
    --camera-url http://192.168.1.10:8000 \
    --task "pick up the red block and place it in the box"
```

Drive the real arms (prompts for confirmation):

```bash
python3 robot_inference_client.py --execute \
    --camera-url http://192.168.1.10:8000 \
    --camera-ids <serial0> <serial1> <serial2> <serial3> \
    --metadata ../gr00t-server/metadata.json \
    --task "pick up the red block and place it in the box"
```

Or use the launcher, which sets up the port-forward for you:

```bash
./run.sh            # dry run
./run.sh --execute  # move the arms
```

## ☸️ In-cluster deployment (GitOps)

The client also ships as a Helm-managed Deployment in the `gr00t-inference`
chart (`templates/robot-client.yaml`, values key `robotClient`). It runs on the
`tegra-ubuntu` node, reaches the policy server via the in-cluster Service
(`gr00t-gr00t-inference:5555`), proxies cameras from the hostNetwork streamer
via the downward-API `HOST_IP`, and egresses to the robot subnet through the
node.

Build and push the image (on the arm64 host so the arch matches), then let Flux
reconcile:

```bash
./build_and_push.sh 0.1
# set robotClient.image.tag in the HelmRelease, commit, push -> Flux deploys
```

By default the chart deploys in **dry run** (`robotClient.execute: false`). To
actually drive the arms, set both `execute: true` and `assumeYes: true` in the
HelmRelease and let Flux reconcile — there is no TTY in-cluster, so `assumeYes`
deliberately bypasses the interactive confirmation. Watch it with:

```bash
kubectl logs -n default deploy/gr00t-gr00t-inference-robot-client -f
```

The registry is parameterized: `build_and_push.sh` reads `${REGISTRY}` (default
`immitationlearning.azurecr.io`) so the image can target any registry.

## ⚙️ Configuration you MUST verify per rig

These determine whether the robot moves correctly, and the exact values are
**not fully derivable from this repo** — confirm them against your recording
setup:

- **`--camera-ids`**: the four `ur-camera-streamer` camera ids in
  `color_0,color_1,color_2,color_3` order. The recorded `modality.json` order is
  `cam_high, cam_low, cam_left_wrist, cam_right_wrist`, but the streamer reports
  cameras by **serial**, and the serial→`color_N` mapping lives in the rig's
  `config_v3.yaml` (not in this repo). Run `--once` and check the dashboard to
  match serials to physical positions before executing. A wrong order feeds the
  policy the wrong views and produces wrong motion.
- **`--robot1-ip` / `--robot2-ip`**: `robot1` is the left follower
  (`192.168.1.80`), `robot2` the right (`192.168.1.90`) per the recorder
  config. Swap if your wiring differs.
- **`--task`**: must match the language annotation used during training.
- **`--control-hz`**: action playback rate. The 16-step horizon implies ~15 fps
  training cadence (default `15`), though some session metadata reports 30.

## 🔑 Key options

Run `python3 robot_inference_client.py --help` for the full list. Most useful:

| Option             | Default | Purpose                                                      |
|--------------------|---------|--------------------------------------------------------------|
| `--execute`        | off     | Actually move the arms (else dry run).                       |
| `--exec-steps`     | 8       | Steps of each 16-step chunk to run before re-querying.       |
| `--control-hz`     | 15      | Action playback rate (match training fps).                   |
| `--max-joint-step` | 0.03    | Per-step joint speed cap (rad).                              |
| `--start-threshold`| 0.30    | First-pose safety gate (rad).                                |
| `--metadata`       | none    | Enable absolute joint clamping from `metadata.json`.         |

## ⚠️ Limitations

- **Grippers are not commanded.** They were constant in the training data and
  excluded from the model's action space.
- **Timing is best-effort.** `servoJ` streams at `--control-hz`; the policy
  query latency determines how often new chunks arrive. Tune `--exec-steps`.
- This client assumes the deployed checkpoint's modality contract
  (`video.color_0..3`, `state.robot{1,2}_arm`, 16-step action horizon). If the
  model is retrained with a different layout, update the constants at the top of
  `robot_inference_client.py`.

## 🧪 Tests

Behavior tests for the ZMQ wire-format encode/decode and the safety clamps live
in [`../tests/test_gr00t_robot_client.py`](../tests/test_gr00t_robot_client.py).
They mock `zmq`, `requests` and `msgpack`, so no hardware or cluster is needed:

```bash
pytest fleet-deployment/inference/tests/test_gr00t_robot_client.py
```
