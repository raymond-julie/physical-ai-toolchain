---
title: Setting Up an On-Prem OSMO Cluster
description: End-to-end tutorial for deploying and accessing the new on-prem OSMO control plane, worker nodes, and management scripts
author: Microsoft Robotics-AI Team
ms.date: 2026-06-11
ms.topic: tutorial
keywords:
  - on-prem osmo
  - cluster setup
  - access
  - deployment
estimated_reading_time: 30
sidebar_position: 1
---

Use the new on-prem OSMO projects in the order they were designed: define inventory, provision prerequisites, initialize the control plane, join workers, deploy OSMO, then use the access scripts for day-two connectivity and troubleshooting.

## 📋 Prerequisites

| Requirement | Details |
| --- | --- |
| Linux nodes | One control plane and one or more workers |
| Windows or Linux management host | Needed for the orchestration entrypoints |
| SSH access | Passwordless or key-based access to every node |
| Kubernetes familiarity | Needed for day-two validation and recovery |
| Network plan | Replace all example placeholders such as `192.168.1.x` with the actual node and robot IPs for your environment |

## 🗂️ Understand the Two On-Prem Project Areas

| Directory | Purpose |
| --- | --- |
| `infrastructure/setup/onprem-osmo/deploy` | Builds and deploys the cluster and OSMO services |
| `infrastructure/setup/onprem-osmo/access` | Creates connectivity, tunneling, diagnostics, and watch helpers |

Treat `deploy` as day-zero and `access` as day-one and day-two operations.

## 🚀 Step 1: Populate the Inventory

Start from the example environment file in `deploy/config` and replace every placeholder IP or hostname with the real values for your cell.

Capture at least:

1. Control-plane hostname and IP.
2. Worker hostnames and IPs.
3. Per-node SSH user.
4. Worker labels.

Do not leave `192.168.1.x` placeholders in the final inventory.

## 🔧 Step 2: Use the PowerShell Orchestrator from the Management Host

The main control point is `deploy.ps1`.

Common entrypoints:

```powershell
cd infrastructure\setup\onprem-osmo\deploy
.\deploy.ps1 -Action all
.\deploy.ps1 -Action status
.\deploy.ps1 -Action gpu-status
```

Use narrow actions while bringing the cluster up the first time:

```powershell
.\deploy.ps1 -Action prerequisites
.\deploy.ps1 -Action control-plane
.\deploy.ps1 -Action workers
.\deploy.ps1 -Action osmo
```

That staged flow makes failures much easier to isolate than a single full deployment attempt.

## 🧱 Step 3: Understand the Deployment Script Order

Inside `deploy/scripts`, the scripts are intentionally numbered.

| Script | Role |
| --- | --- |
| `00-prerequisites.sh` | Base packages and host preparation |
| `01-init-control-plane.sh` | Initialize the Kubernetes control plane |
| `02-join-worker.sh` | Join additional workers |
| `03-deploy-osmo.sh` | Install the OSMO platform |
| `04-install-cli.sh` | Install the CLI locally |
| `05-cleanup.sh` | Remove installed components |
| `06-deploy-local.sh` | Local deployment variant |
| `07-add-node.sh` | Add a node after the cluster already exists |

Do not skip directly to `03-deploy-osmo.sh` on a fresh environment.

## 🔐 Step 4: Establish Access Workflows

After the cluster exists, switch to `access/`.

Important entrypoints:

| Script | Purpose |
| --- | --- |
| `setup-ssh-key.sh` | Provision the Arc SSH keypair used by the access flow |
| `connect.sh` | Open an interactive or one-shot Arc SSH session |
| `01-create-vnet.sh` | Create the network path for remote access |
| `03-tunnel-osmo.sh` | Tunnel the OSMO endpoint |
| `05-diagnose-osmo.sh` | Run connectivity diagnostics |
| `06-watch-cluster.sh` | Watch multiple node endpoints for health |

Use `connect.sh` first to confirm the Arc path works before you debug anything else.

## 🧪 Step 5: Validate the Cluster Before Using It

Validation should happen in layers:

1. SSH to every node.
2. Kubernetes control plane healthy.
3. Worker nodes joined.
4. OSMO services reachable.
5. CLI installed and usable.

If you need to add a node later, use the dedicated add-node path instead of replaying the whole bootstrap:

```powershell
.\deploy.ps1 -Action add-node -NodeIp 192.168.1.x
```

## 🔍 Step 6: Use the Troubleshooting Helpers Instead of Ad Hoc Commands

The new project already contains targeted helpers for day-two operations. Use them first:

1. `validate-inventory.sh` when the environment file looks suspicious.
2. `preflight-check.sh` before rerunning cluster install steps.
3. `smoke-test.yaml` and `smoke-acsa-mirror.sh` for functional validation.
4. `05-diagnose-osmo.sh` and `06-watch-cluster.sh` for remote diagnostics.

That gives you a repeatable support path instead of one-off shell sessions.

## ✅ Verification Checklist

You have the on-prem stack working when:

1. Inventory values are real, not placeholder addresses.
2. `deploy.ps1 -Action status` reports a healthy cluster.
3. Workers are joined with the expected labels.
4. The OSMO service is reachable through the access tooling.
5. Add-node and cleanup workflows are understood before you need them.

## 🔗 Related Documentation

- [Recipes hub](../README.md)
- [Infrastructure README](../../../infrastructure/README.md)
