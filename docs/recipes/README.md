---
title: Recipes
description: Step-by-step guides for the main training, data collection, edge inference, and infrastructure workflows in the physical-ai-toolchain
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: overview
keywords:
	- recipes
	- tutorials
	- data collection
	- training
	- edge inference
	- infrastructure
estimated_reading_time: 5
---

Step-by-step guides that take you from a standing start to a working result. Each recipe is self-contained with prerequisites, runnable commands, and verification steps.

> [!NOTE]
> Recipes assume deployed infrastructure. Complete the [Quickstart](../getting-started/quickstart.md) first if you have not provisioned Azure resources.

## 🚀 Pick a Recipe

| Goal                                          | Recipe                                                                                | Time   |
|-----------------------------------------------|---------------------------------------------------------------------------------------|--------|
| Train an RL policy                            | [Your First RL Training Job](training/your-first-rl-training-job.md)                  | 30 min |
| Train a LeRobot policy                        | [Your First LeRobot Training Job](training/your-first-lerobot-training-job.md)        | 30 min |
| Run the full train → eval → register pipeline | [End-to-End LeRobot Pipeline](training/end-to-end-lerobot-pipeline.md)                | 60 min |
| Operate the new edge capture projects         | [Operating Edge Capture Projects](data-collection/operating-edge-capture-projects.md) | 35 min |
| Configure edge recording                      | [Configuring Edge Data Recording](data-collection/configuring-edge-data-recording.md) | 20 min |
| Prepare a dataset for training                | [Preparing Datasets for Training](data-collection/preparing-datasets-for-training.md) | 30 min |
| Train and evaluate VLA models                 | [Operating VLA Training and Evaluation](training/operating-vla-training-and-evaluation.md) | 28 min |
| Deploy the GR00T edge stack                   | [Deploying the GR00T Edge Inference Stack](edge-inference/deploying-gr00t-edge-inference.md) | 45 min |
| Set up on-prem OSMO                           | [Setting Up an On-Prem OSMO Cluster](infrastructure/setting-up-onprem-osmo-cluster.md) | 60 min |

## 📖 Recipe Catalog

### Training

| Recipe                                                                         | Description                                            | Prerequisites                                |
|--------------------------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------|
| [Your First RL Training Job](training/your-first-rl-training-job.md)           | Submit an Isaac Lab RL training job on OSMO with SKRL  | Deployed infrastructure, OSMO running        |
| [Your First LeRobot Training Job](training/your-first-lerobot-training-job.md) | Submit a LeRobot behavioral cloning job on OSMO        | Deployed infrastructure, HuggingFace dataset |
| [End-to-End LeRobot Pipeline](training/end-to-end-lerobot-pipeline.md)         | Orchestrate train → evaluate → register in one command | Completed basic LeRobot recipe               |
| [Operating VLA Training and Evaluation](training/operating-vla-training-and-evaluation.md) | Train and evaluate GR00T and openpi VLA checkpoints | Prepared VLA dataset, GPU capacity           |

### Data Collection

| Recipe                                                                                | Description                                                         | Prerequisites           |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------------|-------------------------|
| [Operating Edge Capture Projects](data-collection/operating-edge-capture-projects.md) | Use camera_streamer, URCap, dual_recorder, episode_recorder, leader_follower, and dataset tools together | Robot or camera lab access |
| [Configuring Edge Data Recording](data-collection/configuring-edge-data-recording.md) | Set up ROS 2 edge recording on Jetson with chunking and compression | Jetson device, ROS 2    |
| [Preparing Datasets for Training](data-collection/preparing-datasets-for-training.md) | Download, inspect, and validate datasets for LeRobot training       | Python 3.12+, Azure CLI |

### Edge inference

| Recipe | Description | Prerequisites |
| --- | --- | --- |
| [Deploying the GR00T Edge Inference Stack](edge-inference/deploying-gr00t-edge-inference.md) | Build, deploy, and operate the GR00T runtime, UI, robot client, and GitOps overlay | Jetson cluster, registry, model artifact |

### Infrastructure

| Recipe | Description | Prerequisites |
| --- | --- | --- |
| [Setting Up an On-Prem OSMO Cluster](infrastructure/setting-up-onprem-osmo-cluster.md) | Deploy the on-prem OSMO control plane, workers, and access tooling | Management host, Linux nodes, SSH access |

## 🔗 Related Documentation

- [Getting Started](../getting-started/README.md) — infrastructure deployment and first training job
- [Training Guide](../training/README.md) — reference documentation for RL and IL workflows
- [Data Pipeline](../data-pipeline/README.md) — edge recording configuration reference
- [Scripts Reference](../reference/scripts.md) — CLI parameter tables for all submission scripts

<!-- markdownlint-disable MD036 -->
*🤖 Crafted with precision by ✨Copilot following brilliant human instruction,
then carefully refined by our team of discerning human reviewers.*
<!-- markdownlint-enable MD036 -->
