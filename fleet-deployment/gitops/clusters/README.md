# Cluster Overlays

Per-cluster Kustomize overlays that customize fleet deployment manifests for individual edge clusters or cluster groups.

## 🗂️ Clusters

| Cluster | Platform | Description |
| --- | --- | --- |
| [tegra/](tegra/) | Single-node k3s on Jetson AGX Orin | GR00T N1.5 dual-arm UR5e inference stack (server, control UI, robot client, camera streamer, blob sync, GPU device plugin) |

Each cluster directory holds Flux-reconciled manifests: a `flux-system/` overlay
(Flux controllers + the Git sync) plus one subdirectory per workload, each with
its own `kustomization.yaml`. Flux's path `Kustomization` reconciles those
subdirectories directly, so there is no top-level aggregate `kustomization.yaml`.
See [tegra/README.md](tegra/README.md) for the bootstrap steps and the GitRepository
URL the operator must set.


> [!NOTE]
> Replace all example IP placeholders (for example, 192.168.1.x) with the actual robot IP addresses for your environment before running.
