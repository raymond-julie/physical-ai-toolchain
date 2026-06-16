# gr00t-inference Helm Chart

Serves the NVIDIA Isaac-GR00T N1.5 dual-arm UR5e policy on a k3s / Jetson Orin
cluster. The chart deploys the ZMQ inference server, an optional operator control
UI, and the optional closed-loop robot client — all from a single release.

Model weights are **not** baked into any image. They live in an OCI *model
artifact* (CNCF ModelPack, `safetensors`) in a container registry and are pulled
into a persistent volume by a `modctl` init container on pod start.

## 📦 What it deploys

| Component | Template | Enabled by | Purpose |
| --- | --- | --- | --- |
| Inference server | `deployment.yaml`, `service.yaml`, `pvc.yaml` | always | GR00T N1.5 ZMQ policy server (port 5555) + weights cache PVC |
| Weight fetch | init container in `deployment.yaml` | always | `modctl pull` the model artifact into the PVC |
| Data config | `data-config.yaml` | `dataConfigModule.enabled` | Mounts `Ur5eDualArmDataConfig` so N1.5 can rebuild the modality transform |
| Metadata inject | `data-config.yaml` + init container | `model.metadata.inject` | Supplies `experiment_cfg/metadata.json` normalization stats |
| Control UI | `control-ui.yaml`, `control-ui-rbac.yaml` | `controlUi.enabled` | Start/Stop/Status web app (scales the server 0↔1) |
| Robot client | `robot-client.yaml` | `robotClient.enabled` | Closed-loop client that drives the two follower arms |

## 🔧 The modctl weight init container

The weights are a CNCF ModelPack artifact whose layers carry
`org.cncf.model.filepath` rather than `org.opencontainers.image.title`. A plain
`oras pull` skips those layers and leaves the directory empty, so the chart uses
a `modctl` init container (`fetch-weights`) that runs:

```sh
modctl pull <registry>/<model.repository>:<model.tag> \
  --extract-dir /models/gr00t --extract-from-remote
```

The pulled weights land on a `ReadWriteOnce` PVC (`helm.sh/resource-policy: keep`)
so a failed-release uninstall never wipes the ~7 GiB cache. When
`model.tag: latest` and `modctl.alwaysPull: true`, the init container re-pulls on
every start; with a pinned tag and `alwaysPull: false` it skips the pull once the
cache is populated. The `modctl` CLI image is built from
`fleet-deployment/inference/modctl-init/`.

> [!NOTE]
> GR00T N1.5 reads `experiment_cfg/metadata.json` (normalization statistics) from
> the model path and errors if it is absent. This checkpoint's artifact omits it,
> so set `model.metadata.inject: true` to have the init container copy
> `files/metadata.json` into the weights cache.

## ⚙️ Key values

| Value | Default | Description |
| --- | --- | --- |
| `registry` | `immitationlearning.azurecr.io` | **Environment-specific** registry for images and the model artifact. Override per cluster. |
| `imagePullSecret` | `acr-immitationlearning` | Pre-existing `dockerconfigjson` Secret used for image pulls and ORAS/modctl auth |
| `model.repository` / `model.tag` | `gr00t-n15-teradyne-dual-arm` / `latest` | Model artifact location and tag (resolved relative to `registry`) |
| `image.repository` / `image.tag` | `…/gr00t-inference-server` / `TODO` | Serving image; set `tag` to the value you built and pushed |
| `embodimentTag` | `new_embodiment` | Embodiment the checkpoint was trained with |
| `dataConfig` / `dataConfigModule.*` | `""` / disabled | Custom `module:ClassName` data config mounted into the server WORKDIR |
| `denoisingSteps` | `4` | Diffusion denoising steps at inference |
| `gpu.count` / `runtimeClassName` | `1` / `""` | GPU request; set `runtimeClassName: nvidia` on k3s/Jetson |
| `controlUi.enabled` | `false` | Deploy the Start/Stop control UI + least-privilege RBAC |
| `robotClient.enabled` | `false` | Deploy the closed-loop robot client (see safety below) |

See [values.yaml](values.yaml) for the full, commented value set.

## ⚠️ Robot client safety

> [!WARNING]
> The robot client is the closed loop that **physically drives the two follower
> arms**. It defaults to a dry run (`robotClient.execute: false`): it connects,
> queries the policy and logs predicted actions but does **not** move the robots.
> Setting **both** `robotClient.execute: true` and `robotClient.assumeYes: true`
> makes the arms move on deploy. Only enable both deliberately, with the
> workspace clear and an e-stop within reach.

Even in execute mode the per-step clamp (`maxJointStep`) and first-pose start gate
(`startThreshold`) stay active. The client also clamps targets to the trained
absolute joint range when `model.metadata.inject` and `robotClient.metadata.enabled`
are set.

## 🚀 Usage

This chart is normally deployed by Flux via a `HelmRelease`, not by hand — see
[../../clusters/tegra/README.md](../../clusters/tegra/README.md). For local
testing:

```bash
helm install gr00t fleet-deployment/gitops/charts/gr00t-inference \
  --namespace default \
  --set image.tag=0.2-l4t \
  --set runtimeClassName=nvidia
```
