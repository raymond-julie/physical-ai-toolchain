---
title: Deploying the GR00T Edge Inference Stack
description: End-to-end tutorial for building, deploying, and operating the GR00T server, control UI, robot client, modctl init image, and tegra GitOps overlay
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: tutorial
keywords:
  - gr00t
  - edge inference
  - gitops
  - jetson
  - control ui
  - robot client
estimated_reading_time: 30
sidebar_position: 1
---

Deploy the new edge inference stack in the same order the system itself depends on: build the images, confirm the model artifact contract, configure the GitOps values, deploy to the `tegra` cluster, then validate the server and robot client separately before you enable any physical motion.

## 📋 Prerequisites

| Requirement | Details |
| --- | --- |
| Container registry | Push access for the runtime images and model artifact references |
| Cluster | Jetson or compatible Kubernetes node with GPU runtime configured |
| Flux | Working GitOps bootstrap for the target cluster |
| Cameras | `ur-camera-streamer` deployed or otherwise reachable |
| Robot network | Replace all `192.168.1.x` placeholders with the actual follower-arm and streamer IPs for your rig |

## 🧱 Understand the Stack Before Deploying It

The new edge stack is made of distinct projects:

| Project | Purpose |
| --- | --- |
| `gr00t-server` | Runs the ZMQ inference service for the GR00T checkpoint |
| `modctl-init` | Pulls the model artifact into the mounted cache volume |
| `gr00t-control-ui` | Starts and stops the server deployment and shows observability panels |
| `gr00t-robot-client` | Reads cameras and robot joints, asks the policy for actions, and streams targets back to the arms |
| `fleet-deployment/gitops/charts/gr00t-inference` | Helm chart for the combined deployment |
| `fleet-deployment/gitops/clusters/tegra` | Cluster-specific GitOps overlay |

Do not treat the control UI as the robot driver. The control UI only scales the inference deployment. The robot client is the component that can physically move the arms.

## 🚀 Step 1: Build the Runtime Images

Build each component on the correct architecture, usually arm64 for the Jetson target.

```bash
cd fleet-deployment/inference/gr00t-server
./build_and_push.sh 0.1-l4t

cd ../modctl-init
./build_and_push.sh 0.1

cd ../gr00t-control-ui
./build_and_push.sh 0.2

cd ../gr00t-robot-client
./build_and_push.sh 0.2
```

Use `${REGISTRY}` overrides when your registry is not the example ACR.

## 🧪 Step 2: Verify the Model Artifact Contract

Before Helm ever pulls the weights, confirm the model artifact exists and matches the runtime expectations:

1. The artifact is stored in the configured OCI registry.
2. The checkpoint is a GR00T N1.5 checkpoint.
3. The embodiment tag matches the configured value.
4. The data config module matches the training-time layout.

The chart can inject `metadata.json` when the artifact does not contain `experiment_cfg/metadata.json`. Keep that enabled for checkpoints that need the normalization stats.

## ⚙️ Step 3: Set the Helm Values Deliberately

Review these values before you reconcile the cluster:

| Area | Critical values |
| --- | --- |
| Server image | `image.repository`, `image.tag` |
| Model artifact | `registry`, `model.repository`, `model.tag` |
| Runtime | `runtimeClassName`, `gpu.count`, `nodeSelector` |
| Data config | `dataConfig`, `dataConfigModule.*` |
| Control UI | `controlUi.enabled`, service type, node pinning |
| Robot client | `robotClient.enabled`, `cameraIds`, `robot1Ip`, `robot2Ip`, `task` |
| Motion safety | `robotClient.execute`, `robotClient.assumeYes`, `maxJointStep`, `startThreshold` |

Keep `robotClient.execute: false` and `robotClient.assumeYes: false` until the rest of the deployment is verified.

## ☸️ Step 4: Reconcile the `tegra` Overlay

Once the values are correct, update the overlay and let Flux reconcile it.

Watch the rollout:

```bash
flux get kustomizations
flux get helmreleases -n default
kubectl get pods -A
```

The `fetch-weights` init container can take many minutes on a cold cache. That is expected.

## 🔍 Step 5: Validate Each Layer Separately

### Validate the server

1. Confirm the GR00T pod completes its weight pull and reaches ready state.
2. Port-forward the service if needed.
3. Use the robot client's `--ping` or `--once` mode to confirm the server answers.

### Validate the control UI

1. Port-forward the UI service.
2. Confirm the Start and Stop actions change deployment scale.
3. Confirm camera panels and CPU or memory graphs populate.

### Validate the robot client in dry run

Run or inspect the client with `execute` disabled first. Confirm that it:

1. Connects to the streamer.
2. Resolves the four camera ids in the expected `color_0..3` order.
3. Reads both follower arms.
4. Queries the policy and logs actions.

Only after that should you enable execution.

## 🛡️ Step 6: Enable Physical Motion Deliberately

When you are satisfied with the dry-run checks:

1. Set `robotClient.execute: true`.
2. Set `robotClient.assumeYes: true` only if this must run unattended in-cluster.
3. Reconcile again.
4. Keep the workspace clear and the e-stop within reach.

This is the point where the system becomes a real robot-control path, not just an inference demo.

## ✅ Verification Checklist

The edge stack is healthy when:

1. The server pod is ready.
2. The model cache exists on the mounted volume.
3. The control UI can scale the deployment.
4. The robot client can run in dry mode end to end.
5. Motion safety gates are configured before execute mode is enabled.

## 🔗 Related Documentation

- [Inference Runtime](../../../fleet-deployment/inference/README.md)
- [tegra Cluster](../../../fleet-deployment/gitops/clusters/tegra/README.md)
- [GR00T Robot Client README](../../../fleet-deployment/inference/gr00t-robot-client/README.md)
