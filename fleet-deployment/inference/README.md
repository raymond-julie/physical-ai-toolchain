# Inference Runtime

On-device inference code for serving trained robot policies at the edge. It spans
two families: a ROS 2 ACT node, and the ZMQ-based GR00T N1.5 dual-arm stack
(policy server, weight-fetch init image, operator UI, and closed-loop robot
client) that the [GitOps](../gitops) `tegra` cluster deploys to a Jetson Orin.

## 📦 Components

| Component                                      | Kind        | Purpose                                                                                          |
|------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------|
| [`act_inference_node.py`](act_inference_node.py) | ROS 2 node  | ACT policy inference node for ROS 2.                                                              |
| [`gr00t-server/`](gr00t-server)                | Image       | GR00T N1.5 ZMQ policy inference server (`:5555`); weights mounted by the init container, not baked. |
| [`modctl-init/`](modctl-init)                  | Image       | `modctl` CLI init image that pulls the CNCF ModelPack weight artifacts `oras` cannot materialize. |
| [`gr00t-control-ui/`](gr00t-control-ui)        | Image       | FastAPI Start/Stop + observability UI; scales the inference Deployment 0↔1.                       |
| [`gr00t-robot-client/`](gr00t-robot-client)    | Image       | Closed-loop dual-arm UR5e client (ZMQ policy + RTDE `servoJ`) with conservative safety clamps.    |
| [`ur_edge/`](ur_edge)                          | Package     | On-device UR edge runtime — GR00T / SmolVLA / ACT model runners and replay tools (added separately). |

> [!WARNING]
> In the [GitOps](../gitops) `tegra` cluster overlay, the HelmRelease deploys
> `gr00t-robot-client` with `robotClient.execute: true` and `assumeYes: true`,
> so **the arms move automatically** once Flux reconciles — there is no
> interactive confirmation in-cluster. Keep an e-stop within reach and the
> workspace clear before enabling it.

## 🔗 How the GR00T stack fits together

```text
modctl-init ──(pull weights)──► /models/gr00t (PVC)
                                      │
gr00t-control-ui ──(scale 0↔1)──► gr00t-server (ZMQ :5555)
                                      ▲
gr00t-robot-client ──(observe → get_action → servoJ)──┘ ──► UR5e arms
```

The `gr00t-server`, `modctl-init`, `gr00t-control-ui`, and `gr00t-robot-client`
images are built by their per-component `build_and_push.sh` (registry
parameterized via `${REGISTRY}`) and deployed together by the
[`gr00t-inference`](../gitops/charts/gr00t-inference) Helm chart.

## 🧪 Tests

`tests/` holds the root-discovered behavior suites (`act_inference_node`, plotting
and robot-types property tests, and the GR00T client wire-format/clamp tests).
The per-image components also ship focused, hardware-free suites in their own
`tests/` (`gr00t-robot-client/tests`, `gr00t-control-ui/tests`) that stub the
hardware, cluster, and transport dependencies via a local `conftest.py`.


> [!NOTE]
> Replace all example IP placeholders (for example, 192.168.1.x) with the actual robot IP addresses for your environment before running.
