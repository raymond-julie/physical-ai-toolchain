# GR00T N1.5 Inference Server

The runtime image that serves the Isaac-GR00T **N1.5** dual-arm UR5e policy over
ZMQ (port `5555`). It is **only the runtime** — the model weights are not baked
into the image. The Helm chart's `modctl` init container pulls them from the ACR
model artifact and mounts them at `/models/gr00t` on every pod start.

## 🧠 What it does

Runs Isaac-GR00T N1.5's ZMQ inference service
(`scripts/inference_service.py --server`). The chart supplies `--model-path`,
`--embodiment-tag`, `--data-config`, `--denoising-steps`, `--host` and `--port`;
the server loads the mounted checkpoint and answers `get_action` requests from
the [GR00T robot client](../gr00t-robot-client) over a ZMQ REQ/REP socket.

> [!NOTE]
> The image is pinned to the Isaac-GR00T `n1.5-release` tag on purpose. The
> deployed checkpoint declares `model_type: gr00t_n1_5` (see `config.json`),
> which the newer N1.6/N1.7 code on Isaac-GR00T `main` no longer registers — a
> `main`-based image cannot load it.

## 🧱 Image

| Aspect        | Value                                                                  |
|---------------|------------------------------------------------------------------------|
| Base          | `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (JetPack 6.2 / L4T r36.4, arm64)   |
| GR00T         | `gr00t[orin]` from the jetson-ai-lab wheel index, pinned `n1.5-release` |
| From source   | cuDSS 0.6.0, `pytorch3d==0.7.8` (CUDA ops, `TORCH_CUDA_ARCH_LIST=8.7`), decord + ffmpeg n4.4.2 |
| Entry point   | `python3 scripts/inference_service.py` (chart supplies the server args) |
| Exposed port  | `5555` (ZMQ REQ/REP, not HTTP)                                          |
| Model weights | **Not baked** — mounted at `/models/gr00t` by the modctl init container |

## 📦 Reference artifacts

These ship with the image and are consumed by the chart, not by the build:

| File            | Purpose                                                                                          |
|-----------------|--------------------------------------------------------------------------------------------------|
| `config.json`   | GR00T N1.5 model config (action_dim 32, action_horizon 16, Eagle/Qwen3-1.7B + Siglip2 backbone, diffusion action head). Reference copy of the checkpoint config. |
| `metadata.json` | Per-joint normalization statistics for `new_embodiment` (robot1/robot2 arm + gripper). GR00T N1.5 requires `experiment_cfg/metadata.json`; the chart injects this copy, and the robot client reads it for absolute action clamping. |

## 🚀 Build

Build on the arm64/L4T host so the image arch matches the Jetson cluster (the
Orin extras compile from source, so the build is slow):

```bash
./build_and_push.sh 0.1-l4t
```

The registry is parameterized: `build_and_push.sh` reads `${REGISTRY}` (default
`immitationlearning.azurecr.io`) so the image can target any registry. Optional
positional args override the base image and the Isaac-GR00T ref:

```bash
REGISTRY=myregistry.azurecr.io ./build_and_push.sh 0.1-l4t nvcr.io/nvidia/l4t-jetpack:r36.4.0 n1.5-release
```

## ☸️ In-cluster deployment (GitOps)

Deployed by the `gr00t-inference` Helm chart
([`fleet-deployment/gitops/charts/gr00t-inference`](../../gitops/charts/gr00t-inference)).
The chart wires the `fetch-weights` modctl init container, the data-config
ConfigMap, the `metadata.json` injection, and the ZMQ Service. See the chart and
the `tegra` cluster overlay for the deployed values.
