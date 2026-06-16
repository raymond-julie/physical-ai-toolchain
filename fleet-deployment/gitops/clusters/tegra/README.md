# tegra Cluster

GitOps overlay for **tegra**, a single-node k3s cluster on an NVIDIA Jetson AGX
Orin (`tegra-ubuntu`). Flux reconciles this directory and deploys the full edge
robot-inference stack: the GR00T N1.5 policy server, the operator control UI, the
closed-loop robot client, the camera streamer, the blob uploader, and the NVIDIA
device plugin.

> [!WARNING]
> The `gr00t-inference` HelmRelease in this overlay deploys the robot client with
> `robotClient.execute: true` **and** `robotClient.assumeYes: true`. **The robot
> arms WILL move when Flux reconciles this cluster.** This is intentional for the
> demo rig, but only apply this overlay with the workspace clear, the arms free to
> move, and an e-stop within reach. To deploy without motion, set both back to
> `false` in [gr00t-inference/helmrelease.yaml](gr00t-inference/helmrelease.yaml)
> before bootstrap (the per-step clamp and first-pose start gate stay active in
> execute mode regardless).

## 🗂️ What is in this overlay

| Directory | Namespace | Deploys |
| --- | --- | --- |
| [flux-system/](flux-system/) | `flux-system` | Flux controllers (`gotk-components.yaml`) + the `GitRepository`/`Kustomization` sync (`gotk-sync.yaml`) |
| [gr00t-inference/](gr00t-inference/) | `default` | `HelmRelease` for the local `gr00t-inference` chart (server + control UI + robot client) |
| [ur-camera-streamer/](ur-camera-streamer/) | `default` | hostNetwork MJPEG streamer for the four Orbbec GMSL cameras |
| [blob-sync/](blob-sync/) | `default` | Uploads `recordings_lerobot` episodes to Azure Blob |
| [nvidia-device-plugin/](nvidia-device-plugin/) | `kube-system` | Advertises the Jetson GPU as `nvidia.com/gpu` (Tegra discovery strategy) |

Flux's path `Kustomization` reconciles each subdirectory's own
`kustomization.yaml`; there is intentionally no top-level aggregate
`kustomization.yaml`.

## 🔗 Set the GitRepository URL before bootstrap

[flux-system/gotk-sync.yaml](flux-system/gotk-sync.yaml) ships with a **placeholder**
Git URL:

```yaml
url: ssh://git@github.com/<your-org>/physical-ai-toolchain
```

The operator **must** point this at their own fork/clone of this repository before
bootstrapping, either by:

- passing `--url` to [../../bootstrap.sh](../../bootstrap.sh) (recommended — `flux
  bootstrap` regenerates `gotk-sync.yaml` with your URL and the correct path), or
- editing the `url` field directly and applying the committed manifests.

The Flux sync path has already been rewritten for this monorepo:
`spec.path: ./fleet-deployment/gitops/clusters/tegra`. The `HelmRelease` chart
path is likewise `./fleet-deployment/gitops/charts/gr00t-inference`.

## 🚀 Bootstrap

From a workstation with `kubectl` pointed at the tegra cluster and the `flux` CLI
installed:

```bash
# Recommended: canonical flux bootstrap (regenerates and commits gotk-* manifests).
fleet-deployment/gitops/bootstrap.sh \
  --url ssh://git@github.com/<your-org>/physical-ai-toolchain \
  --branch main \
  --path ./fleet-deployment/gitops/clusters/tegra

# Alternative: apply the manifests already committed in this repo (no commit back).
fleet-deployment/gitops/bootstrap.sh --apply
```

Run `fleet-deployment/gitops/bootstrap.sh --config-preview` to print the resolved
configuration without changing the cluster.

### Required secrets

These Secrets are referenced by the overlay and must exist before reconciliation
succeeds (they are intentionally **not** committed):

| Secret | Namespace | Type | Used by |
| --- | --- | --- | --- |
| `flux-system` | `flux-system` | SSH deploy key | Flux `GitRepository` auth (created by `flux bootstrap`) |
| `acr-immitationlearning` | `default` | `dockerconfigjson` | Image pulls + ORAS/modctl weight auth |
| `blob-sync-secret` | `default` | Opaque (`BLOB_SYNC_CONTAINER_URL` SAS) | `blob-sync` uploader |

### Host prerequisites on `tegra-ubuntu`

- The `nvidia` containerd runtime and `RuntimeClass` registered by k3s.
- `/etc/trainmybot/config_v3.yaml` present (camera serials + arm topology, mounted
  read-only into `ur-camera-streamer`).
- The four Orbbec GMSL cameras wired and visible as `/dev` V4L2 nodes.
- A route from the node to the robot subnet (follower arms `192.168.1.11` /
  `192.168.1.13`).

## ✅ Verify

```bash
flux get kustomizations
flux get helmreleases -n default
kubectl get pods -A
```

The `fetch-weights` init container pulls ~7 GiB via `modctl` (~13 min) on first
start, so the `gr00t-inference` pod stays in `Init` for a while; the HelmRelease
`timeout` is set to 25m to accommodate this.
