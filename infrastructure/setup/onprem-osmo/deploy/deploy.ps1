<#
.SYNOPSIS
  Orchestrates NVIDIA OSMO cluster deployment from a Windows management machine.

.DESCRIPTION
  This script connects to Linux nodes via SSH and executes the deployment
  scripts remotely in the correct order:
    1. Install prerequisites on all nodes
    2. Initialize control plane
    3. Join worker nodes
    4. Deploy OSMO
    5. Install OSMO CLI (local)

.PARAMETER ConfigFile
  Path to the inventory configuration file. Default: config/inventory.env

.PARAMETER Action
  Deployment action to perform:
    all           - Full deployment (default)
    prerequisites - Install prerequisites on all nodes
    control-plane - Initialize control plane only
    workers       - Join worker nodes only
    osmo          - Deploy OSMO on existing cluster
    cli           - Install OSMO CLI locally
    cleanup       - Tear down everything
    status        - Show cluster status
    gpu-status    - Show GPU health and utilization across cluster

.PARAMETER SshKeyPath
  Path to SSH private key. Default: ~/.ssh/id_rsa

.PARAMETER NodeIp
  IP address of a new node to add (used with -Action add-node)

.PARAMETER NodeUser
  SSH user for the new node (used with -Action add-node). Default: ubuntu

.PARAMETER NodeLabels
  Kubernetes labels for the new node (used with -Action add-node). Default: node_group=compute

.EXAMPLE
  .\deploy.ps1 -Action all
  .\deploy.ps1 -Action prerequisites
  .\deploy.ps1 -Action status
  .\deploy.ps1 -Action add-node -NodeIp 192.168.1.104
#>

[CmdletBinding()]
param(
  [Parameter()]
  [string]$ConfigFile = "config\inventory.env",

  [Parameter()]
  [ValidateSet("all", "prerequisites", "control-plane", "workers", "osmo", "cli", "cleanup", "status", "gpu-status", "local", "add-node")]
  [string]$Action = "all",

  [Parameter()]
  [string]$SshKeyPath = $(if (Test-Path "$env:USERPROFILE\.ssh\id_ed25519") { "$env:USERPROFILE\.ssh\id_ed25519" } else { "$env:USERPROFILE\.ssh\id_rsa" }),

  [Parameter()]
  [string]$NodeIp = "",

  [Parameter()]
  [string]$NodeUser = "ubuntu",

  [Parameter()]
  [string]$NodeLabels = "node_group=compute"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

#region Configuration Loading

function Read-InventoryConfig {
  param([string]$Path)

  $configPath = Join-Path $ScriptRoot $Path
  if (-not (Test-Path $configPath)) {
    Write-Error "Configuration file not found: $configPath`nCopy config\inventory.example.env to config\inventory.env and edit it."
    return
  }

  $config = @{}
  $content = Get-Content $configPath -Raw

  # Parse simple KEY=VALUE pairs (skip arrays for manual handling)
  foreach ($line in ($content -split "`n")) {
    $line = $line.Trim()
    if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
    if ($line -match '^\s*([A-Z_]+)\s*=\s*"([^"]*)"') {
      $config[$Matches[1]] = $Matches[2]
    }
  }

  # Parse array values
  $workerIps = @()
  $workerHostnames = @()
  $workerUsers = @()
  $workerLabels = @()

  if ($content -match 'WORKER_IPS=\(([^)]+)\)') {
    $workerIps = ($Matches[1] -replace '"', '').Trim() -split '\s+'
  }
  if ($content -match 'WORKER_HOSTNAMES=\(([^)]+)\)') {
    $workerHostnames = ($Matches[1] -replace '"', '').Trim() -split '\s+'
  }
  if ($content -match 'WORKER_USERS=\(([^)]+)\)') {
    $workerUsers = ($Matches[1] -replace '"', '').Trim() -split '\s+'
  }

  # Parse multi-line WORKER_LABELS array
  $labelMatches = [regex]::Matches($content, 'WORKER_LABELS=\(([\s\S]*?)\)')
  if ($labelMatches.Count -gt 0) {
    $labelBlock = $labelMatches[0].Groups[1].Value
    $workerLabels = [regex]::Matches($labelBlock, '"([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
  }

  $config["WORKER_IPS"] = $workerIps
  $config["WORKER_HOSTNAMES"] = $workerHostnames
  $config["WORKER_USERS"] = $workerUsers
  $config["WORKER_LABELS"] = $workerLabels

  return $config
}

#endregion

#region SSH Helpers

function Invoke-SshCommand {
  param(
    [string]$TargetHost,
    [string]$User,
    [string]$Command,
    [string]$KeyPath = $SshKeyPath,
    [switch]$Quiet,
    # When set, a non-zero exit code from the remote command causes a
    # terminating error. Callers that tolerate failure (e.g. `|| true`
    # patterns) should leave this off.
    [switch]$FailOnError
  )

  if (-not $Quiet) {
    Write-Host "[SSH] ${User}@${TargetHost}: $Command" -ForegroundColor Cyan
  }

  # Normalize to LF — PowerShell here-strings on Windows use CRLF, which bash
  # chokes on for multi-line scripts ("syntax error near unexpected token `$'do\r''").
  $Command = $Command -replace "`r`n", "`n" -replace "`r", "`n"

  $sshArgs = @(
    "-i", $KeyPath,
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    "${User}@${TargetHost}",
    $Command
  )

  # Stream output line-by-line so long-running remote commands (helm --wait,
  # kubeadm init, apt installs) show live progress instead of appearing frozen.
  # Each line is both echoed to the console and emitted to the pipeline so
  # callers that do `$output = Invoke-SshCommand ...` still get the full text.
  $collected = New-Object System.Collections.Generic.List[string]
  & ssh @sshArgs 2>&1 | ForEach-Object {
    $line = [string]$_
    if (-not $Quiet) {
      Write-Host "  $line"
    }
    $collected.Add($line)
  }
  $exit = $LASTEXITCODE
  if ($exit -ne 0) {
    if (-not $Quiet) {
      Write-Warning "SSH command returned exit code $exit"
    }
    if ($FailOnError) {
      throw "Remote command on ${User}@${TargetHost} failed with exit code ${exit}: $Command"
    }
  }

  return $collected.ToArray()
}

function Copy-ScriptsToNode {
  param(
    [string]$TargetHost,
    [string]$User,
    [string]$KeyPath = $SshKeyPath
  )

  Write-Host "[SCP] Copying deployment scripts to ${User}@${TargetHost}..." -ForegroundColor Cyan

  $localScripts = Join-Path $ScriptRoot "scripts"
  $localConfig = Join-Path $ScriptRoot "config"

  # Stage LF-normalized copies (Windows CRLF breaks bash/kubeadm scripts on Linux)
  $staging = Join-Path ([IO.Path]::GetTempPath()) ("osmo-stage-" + [Guid]::NewGuid().ToString("N"))
  $stagedScripts = Join-Path $staging "scripts"
  $stagedConfig  = Join-Path $staging "config"
  New-Item -ItemType Directory -Path $stagedScripts -Force | Out-Null
  New-Item -ItemType Directory -Path $stagedConfig  -Force | Out-Null
  foreach ($src in Get-ChildItem -Path $localScripts -File) {
    $text = [IO.File]::ReadAllText($src.FullName) -replace "`r`n", "`n"
    [IO.File]::WriteAllText((Join-Path $stagedScripts $src.Name), $text)
  }
  foreach ($src in Get-ChildItem -Path $localConfig -File) {
    $text = [IO.File]::ReadAllText($src.FullName) -replace "`r`n", "`n"
    [IO.File]::WriteAllText((Join-Path $stagedConfig $src.Name), $text)
  }

  # Create remote directory
  Invoke-SshCommand -TargetHost $TargetHost -User $User -Command "mkdir -p ~/osmo-deployment/scripts ~/osmo-deployment/config" -KeyPath $KeyPath

  # Copy scripts
  $scpArgs = @(
    "-i", $KeyPath,
    "-o", "StrictHostKeyChecking=no",
    "-r",
    "${stagedScripts}/.",
    "${User}@${TargetHost}:~/osmo-deployment/scripts/"
  )
  & scp @scpArgs 2>&1 | Out-Null

  # Copy config
  $scpArgs = @(
    "-i", $KeyPath,
    "-o", "StrictHostKeyChecking=no",
    "-r",
    "${stagedConfig}/.",
    "${User}@${TargetHost}:~/osmo-deployment/config/"
  )
  & scp @scpArgs 2>&1 | Out-Null

  Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue

  # Make scripts executable
  Invoke-SshCommand -TargetHost $TargetHost -User $User -Command "chmod +x ~/osmo-deployment/scripts/*.sh" -KeyPath $KeyPath

  Write-Host "[OK] Scripts copied to ${TargetHost}" -ForegroundColor Green
}

#endregion

#region Deployment Actions

function Install-Prerequisites {
  param($Config)

  Write-Host "`n=== Installing Prerequisites on All Nodes ===" -ForegroundColor Yellow

  # Control plane
  $cpHost = $Config["CONTROL_PLANE_IP"]
  $cpUser = $Config["CONTROL_PLANE_USER"]
  Write-Host "`nControl Plane: $cpHost" -ForegroundColor Magenta

  Copy-ScriptsToNode -TargetHost $cpHost -User $cpUser
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -FailOnError -Command "sudo bash ~/osmo-deployment/scripts/00-prerequisites.sh" | Out-Null

  # Workers
  $workerIps = $Config["WORKER_IPS"]
  $workerUsers = $Config["WORKER_USERS"]

  for ($i = 0; $i -lt $workerIps.Count; $i++) {
    $wHost = $workerIps[$i]
    $wUser = $workerUsers[$i]
    Write-Host "`nWorker $($i + 1): $wHost" -ForegroundColor Magenta

    Copy-ScriptsToNode -TargetHost $wHost -User $wUser
    Invoke-SshCommand -TargetHost $wHost -User $wUser -FailOnError -Command "sudo bash ~/osmo-deployment/scripts/00-prerequisites.sh" | Out-Null
  }

  Write-Host "`n[OK] Prerequisites installed on all nodes." -ForegroundColor Green
}

function Initialize-ControlPlane {
  param($Config)

  Write-Host "`n=== Initializing Control Plane ===" -ForegroundColor Yellow

  $cpHost = $Config["CONTROL_PLANE_IP"]
  $cpUser = $Config["CONTROL_PLANE_USER"]

  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -FailOnError `
    -Command "sudo bash ~/osmo-deployment/scripts/01-init-control-plane.sh --config ~/osmo-deployment/config/inventory.env" | Out-Null

  # Retrieve the join command (-Quiet to avoid streaming noise; the file is
  # the source of truth, not the console).
  Write-Host "`nRetrieving join command..." -ForegroundColor Cyan
  $joinCmd = Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "cat ~/osmo-deployment/config/join-command.sh" -Quiet

  $localJoinFile = Join-Path $ScriptRoot "config\join-command.sh"
  $joinCmd | Out-File -FilePath $localJoinFile -Encoding utf8 -Force

  # Retrieve kubeconfig
  Write-Host "Retrieving kubeconfig..." -ForegroundColor Cyan
  $scpArgs = @(
    "-i", $SshKeyPath,
    "-o", "StrictHostKeyChecking=no",
    "${cpUser}@${cpHost}:~/osmo-deployment/config/kubeconfig",
    (Join-Path $ScriptRoot "config\kubeconfig")
  )
  & scp @scpArgs 2>&1 | Out-Null

  Write-Host "[OK] Control plane initialized." -ForegroundColor Green
}

function Join-Workers {
  param($Config)

  Write-Host "`n=== Joining Worker Nodes ===" -ForegroundColor Yellow

  $workerIps = $Config["WORKER_IPS"]
  $workerUsers = $Config["WORKER_USERS"]
  $workerLabels = $Config["WORKER_LABELS"]

  $joinFile = Join-Path $ScriptRoot "config\join-command.sh"
  if (-not (Test-Path $joinFile)) {
    Write-Error "Join command file not found. Initialize the control plane first."
    return
  }

  $joinCmd = (Get-Content $joinFile -Raw).Trim()

  for ($i = 0; $i -lt $workerIps.Count; $i++) {
    $wHost = $workerIps[$i]
    $wUser = $workerUsers[$i]
    $labels = if ($i -lt $workerLabels.Count) { $workerLabels[$i] } else { "node_group=compute" }

    Write-Host "`nWorker $($i + 1): $wHost (labels: $labels)" -ForegroundColor Magenta

    Invoke-SshCommand -TargetHost $wHost -User $wUser -FailOnError `
      -Command "sudo bash ~/osmo-deployment/scripts/02-join-worker.sh --join-command '$joinCmd' --labels '$labels'" | Out-Null
  }

  # Apply labels from control plane
  $cpHost = $Config["CONTROL_PLANE_IP"]
  $cpUser = $Config["CONTROL_PLANE_USER"]

  Write-Host "`nApplying node labels from control plane..." -ForegroundColor Cyan
  for ($i = 0; $i -lt $workerIps.Count; $i++) {
    # kubelet registers nodes with lowercased hostnames regardless of what
    # `hostname` returns, so always lowercase when passing to kubectl.
    $hostname = $Config["WORKER_HOSTNAMES"][$i].ToLower()
    $labels = if ($i -lt $workerLabels.Count) { $workerLabels[$i] } else { "node_group=compute" }

    Invoke-SshCommand -TargetHost $cpHost -User $cpUser `
      -Command "kubectl label node $hostname $labels --overwrite 2>/dev/null || true"
  }

  Write-Host "[OK] All workers joined." -ForegroundColor Green
}

function Install-Osmo {
  param($Config)

  Write-Host "`n=== Deploying NVIDIA OSMO ===" -ForegroundColor Yellow

  $cpHost = $Config["CONTROL_PLANE_IP"]
  $cpUser = $Config["CONTROL_PLANE_USER"]

  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -FailOnError `
    -Command "bash ~/osmo-deployment/scripts/03-deploy-osmo.sh --config ~/osmo-deployment/config/inventory.env --values ~/osmo-deployment/config/osmo-values.yaml" | Out-Null

  Write-Host "[OK] OSMO deployed." -ForegroundColor Green
}

function Install-OsmoCli {
  Write-Host "`n=== Installing OSMO CLI ===" -ForegroundColor Yellow

  Write-Host "The OSMO CLI is Linux/macOS only." -ForegroundColor Yellow
  Write-Host "To use it from Windows, SSH into the control plane:" -ForegroundColor Yellow
  Write-Host "  ssh <user>@<control-plane-ip>" -ForegroundColor Cyan
  Write-Host "  bash ~/osmo-deployment/scripts/04-install-cli.sh" -ForegroundColor Cyan
  Write-Host ""
  Write-Host "Or use WSL on this Windows machine:" -ForegroundColor Yellow
  Write-Host "  wsl bash scripts/04-install-cli.sh" -ForegroundColor Cyan
}

function Install-LocalOsmo {
  Write-Host "`n=== Local Single-Node OSMO Deployment ===" -ForegroundColor Yellow
  Write-Host ""

  # Check if WSL is available
  $wslCheck = Get-Command wsl -ErrorAction SilentlyContinue
  if (-not $wslCheck) {
    Write-Error @"
WSL (Windows Subsystem for Linux) is required for local deployment.
Install WSL: wsl --install
Then re-run this script.
"@
    return
  }

  # Determine the correct WSL path for our scripts
  $winPath = (Resolve-Path $ScriptRoot).Path
  # Convert Windows path to WSL path
  $wslPath = wsl wslpath -a "$winPath" 2>$null
  if (-not $wslPath) {
    $driveLetter = $winPath.Substring(0, 1).ToLower()
    $wslPath = "/mnt/$driveLetter" + ($winPath.Substring(2) -replace '\\', '/')
  }

  Write-Host "WSL project path: $wslPath" -ForegroundColor Gray
  Write-Host ""

  # Prepare local config if not already present
  $localConfig = Join-Path $ScriptRoot "config\inventory.env"
  $localTemplate = Join-Path $ScriptRoot "config\inventory.local.env"
  if (-not (Test-Path $localConfig)) {
    if (Test-Path $localTemplate) {
      Copy-Item $localTemplate $localConfig
      Write-Host "[OK] Created config\inventory.env from local template." -ForegroundColor Green
    } else {
      Write-Error "No configuration found. Copy config\inventory.local.env to config\inventory.env."
      return
    }
  }

  Write-Host "Running local deployment via WSL..." -ForegroundColor Cyan
  Write-Host "This will install Kubernetes and OSMO on your WSL instance." -ForegroundColor Gray
  Write-Host "You may be prompted for the sudo password in WSL." -ForegroundColor Yellow
  Write-Host ""

  # Build the command with any extra flags
  $deployCmd = "cd '$wslPath' && sudo bash scripts/06-deploy-local.sh --config config/inventory.env --values config/osmo-values.yaml"

  wsl bash -c $deployCmd

  if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[OK] Local OSMO deployment complete!" -ForegroundColor Green
  } else {
    Write-Warning "Local deployment finished with exit code $LASTEXITCODE. Check the output above."
  }
}

function Add-Node {
  param($Config)

  Write-Host "`n=== Adding New Worker Node ===" -ForegroundColor Yellow

  if (-not $NodeIp) {
    Write-Error "Node IP is required. Usage: .\deploy.ps1 -Action add-node -NodeIp <IP> [-NodeUser <USER>] [-NodeLabels <LABELS>]"
    return
  }

  $cpHost = $Config["CONTROL_PLANE_IP"]
  $cpUser = $Config["CONTROL_PLANE_USER"]

  Write-Host "  Control Plane: $cpHost" -ForegroundColor Gray
  Write-Host "  New Node IP:   $NodeIp" -ForegroundColor Gray
  Write-Host "  New Node User: $NodeUser" -ForegroundColor Gray
  Write-Host "  Labels:        $NodeLabels" -ForegroundColor Gray
  Write-Host ""

  # Step 1: Copy scripts to the new node and install prerequisites
  Write-Host "Step 1: Installing prerequisites on $NodeIp..." -ForegroundColor Magenta
  Copy-ScriptsToNode -TargetHost $NodeIp -User $NodeUser
  Invoke-SshCommand -TargetHost $NodeIp -User $NodeUser -Command "sudo bash ~/osmo-deployment/scripts/00-prerequisites.sh" | Out-Null

  # Step 2: Generate a fresh join token from the control plane
  Write-Host "`nStep 2: Generating join token..." -ForegroundColor Magenta
  $joinCmd = Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "sudo kubeadm token create --print-join-command" -Quiet
  $joinCmd = ($joinCmd | Select-Object -Last 1).Trim()
  Write-Host "  Join command obtained." -ForegroundColor Gray

  # Step 3: Join the new node
  Write-Host "`nStep 3: Joining node to cluster..." -ForegroundColor Magenta
  Invoke-SshCommand -TargetHost $NodeIp -User $NodeUser `
    -Command "sudo $joinCmd --cri-socket='unix:///run/containerd/containerd.sock'" | Out-Null

  # Step 4: Apply labels from control plane
  Write-Host "`nStep 4: Applying labels..." -ForegroundColor Magenta
  # Wait a moment for the node to register
  Start-Sleep -Seconds 10

  # Detect the hostname of the new node by IP
  $nodeHostname = Invoke-SshCommand -TargetHost $NodeIp -User $NodeUser -Command "hostname" -Quiet
  $nodeHostname = ($nodeHostname | Select-Object -Last 1).Trim()

  Invoke-SshCommand -TargetHost $cpHost -User $cpUser `
    -Command "kubectl label node $nodeHostname $NodeLabels --overwrite 2>/dev/null || true"

  # Step 5: Show updated cluster status
  Write-Host "`nStep 5: Verifying..." -ForegroundColor Magenta
  Start-Sleep -Seconds 5
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "kubectl get nodes -o wide" | Out-Null

  Write-Host "`n[OK] Node '$nodeHostname' ($NodeIp) added to the cluster." -ForegroundColor Green
  Write-Host "Remember to update config\inventory.env with the new node details." -ForegroundColor Yellow
}

function Invoke-Cleanup {
  param($Config)

  Write-Host "`n=== Cleaning Up OSMO Cluster ===" -ForegroundColor Yellow

  $cpHost = $Config["CONTROL_PLANE_IP"]
  $cpUser = $Config["CONTROL_PLANE_USER"]

  # Ensure each worker has the latest cleanup script, then reset it. We can't
  # assume workers already have ~/osmo-deployment on disk after a partial
  # deploy, so re-push before running.
  $workerIps = $Config["WORKER_IPS"]
  $workerUsers = $Config["WORKER_USERS"]

  for ($i = 0; $i -lt $workerIps.Count; $i++) {
    $wHost = $workerIps[$i]
    $wUser = $workerUsers[$i]
    Write-Host "`nResetting worker $($i + 1): $wHost" -ForegroundColor Magenta
    # A lightweight inline reset — the worker has no Helm/OSMO to remove, just
    # kubeadm + kubelet + CNI state + cached etcd/kubelet dirs.
    $resetCmd = @'
sudo kubeadm reset -f 2>/dev/null || true
sudo systemctl stop kubelet 2>/dev/null || true
# Kill any leftover containerd-shim/pod containers from the previous cluster
# (kubeadm reset doesn't always clean these up).
if command -v crictl >/dev/null 2>&1; then
  sudo crictl rm -af 2>/dev/null || true
  sudo crictl rmp -af 2>/dev/null || true
fi
# Nuke containerd CRI metadata so orphaned sandboxes (stuck because CNI was
# already removed) don't linger.
if systemctl is-active containerd >/dev/null 2>&1; then
  sudo systemctl stop containerd 2>/dev/null || true
  sudo rm -rf /var/lib/containerd/io.containerd.metadata.v1.bolt \
              /var/lib/containerd/io.containerd.runtime.v2.task \
              /var/lib/containerd/io.containerd.sandbox.controller.v1.shim \
              /run/containerd/io.containerd.runtime.v2.task \
              /run/containerd/io.containerd.sandbox.controller.v1.shim 2>/dev/null || true
  sudo systemctl start containerd 2>/dev/null || true
fi
sudo rm -rf /etc/cni/net.d /var/lib/kubelet /var/lib/etcd /etc/kubernetes /var/lib/cni /var/run/kubernetes 2>/dev/null || true
# Unmount Calico cgroup bind-mount before removing /run/calico
for mp in $(mount | awk '/\/run\/calico/ {print $3}'); do sudo umount "$mp" 2>/dev/null || sudo umount -l "$mp" 2>/dev/null || true; done
sudo rm -rf /var/lib/calico /var/log/calico /run/calico /run/flannel /var/log/pods /var/log/containers 2>/dev/null || true
rm -rf ~/.kube ~/.cache/helm ~/.config/helm ~/.local/share/helm 2>/dev/null || true
# Flush AND delete user-defined chains in every table (kube-proxy creates
# KUBE-* chains in filter, nat, and mangle).
for t in filter nat mangle raw; do
  sudo iptables -t $t -F 2>/dev/null || true
  sudo iptables -t $t -X 2>/dev/null || true
done
sudo ipvsadm --clear 2>/dev/null || true
# Stop transient kubepods*.slice systemd units
for s in $(systemctl list-units --all --no-legend 2>/dev/null | awk '/kubepods/ {print $1}'); do
  sudo systemctl stop "$s" 2>/dev/null || true
  sudo systemctl reset-failed "$s" 2>/dev/null || true
done
# Destroy Calico ipsets
if command -v ipset >/dev/null 2>&1; then
  for s in $(sudo ipset list -n 2>/dev/null | grep -E '^cali'); do
    sudo ipset destroy "$s" 2>/dev/null || true
  done
fi
# Delete Calico/bird routes (incl. blackhole pod CIDR)
for r in $(ip route 2>/dev/null | awk '/proto bird/ {print $1" "$2}'); do
  sudo ip route del $r 2>/dev/null || true
done
# Calico ip rule 220 + BGP routing table
sudo ip rule del priority 220 2>/dev/null || true
sudo ip route flush table 220 2>/dev/null || true
# Stale /etc/hosts entries from prior deploys
sudo sed -i '/quick-start\.osmo/d; /\.cluster\.local/d' /etc/hosts 2>/dev/null || true
# Delete Calico/CNI virtual interfaces and unload ipip
for iface in $(ip -br link show 2>/dev/null | awk '/^cali/ {print $1}' | cut -d@ -f1); do
  sudo ip link del "$iface" 2>/dev/null || true
done
# Orphaned veth peers and kube dummy0
for v in $(ip -br link show type veth 2>/dev/null | awk '{print $1}' | cut -d@ -f1 | grep -vE '^(docker|br-)'); do
  sudo ip link del "$v" 2>/dev/null || true
done
for iface in tunl0 vxlan.calico kube-ipvs0 cni0 flannel.1 dummy0; do
  sudo ip link del "$iface" 2>/dev/null || true
done
# Stale CNI network namespaces
for ns in $(ip netns list 2>/dev/null | awk '/^cni-/ {print $1}'); do
  sudo ip netns del "$ns" 2>/dev/null || true
done
sudo modprobe -r ipip 2>/dev/null || true
'@
    Invoke-SshCommand -TargetHost $wHost -User $wUser -Command $resetCmd | Out-Null
  }

  # Control plane last — it tears down Helm releases + kubeadm.
  Write-Host "`nResetting control plane: $cpHost" -ForegroundColor Magenta
  Copy-ScriptsToNode -TargetHost $cpHost -User $cpUser
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser `
    -Command "sudo bash ~/osmo-deployment/scripts/05-cleanup.sh --config ~/osmo-deployment/config/inventory.env" | Out-Null

  # Also wipe kubelet/etcd dirs on the CP after kubeadm reset (05-cleanup.sh
  # runs kubeadm reset but leaves /var/lib/etcd so a subsequent init doesn't
  # fail with "etcd data directory not empty").
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command @'
if command -v crictl >/dev/null 2>&1; then
  sudo crictl rm -af 2>/dev/null || true
  sudo crictl rmp -af 2>/dev/null || true
fi
sudo rm -rf /var/lib/etcd /var/lib/kubelet /etc/kubernetes /var/lib/cni /var/run/kubernetes 2>/dev/null || true
for mp in $(mount | awk '/\/run\/calico/ {print $3}'); do sudo umount "$mp" 2>/dev/null || sudo umount -l "$mp" 2>/dev/null || true; done
sudo rm -rf /var/lib/calico /var/log/calico /run/calico /run/flannel /var/log/pods /var/log/containers 2>/dev/null || true
rm -rf ~/.kube ~/.cache/helm ~/.config/helm ~/.local/share/helm 2>/dev/null || true
for t in filter nat mangle raw; do
  sudo iptables -t $t -F 2>/dev/null || true
  sudo iptables -t $t -X 2>/dev/null || true
done
sudo ipvsadm --clear 2>/dev/null || true
for iface in $(ip -br link show 2>/dev/null | awk '/^cali/ {print $1}' | cut -d@ -f1); do
  sudo ip link del "$iface" 2>/dev/null || true
done
for v in $(ip -br link show type veth 2>/dev/null | awk '{print $1}' | cut -d@ -f1 | grep -vE '^(docker|br-)'); do
  sudo ip link del "$v" 2>/dev/null || true
done
for iface in tunl0 vxlan.calico kube-ipvs0 cni0 flannel.1 dummy0; do
  sudo ip link del "$iface" 2>/dev/null || true
done
for ns in $(ip netns list 2>/dev/null | awk '/^cni-/ {print $1}'); do
  sudo ip netns del "$ns" 2>/dev/null || true
done
sudo modprobe -r ipip 2>/dev/null || true
'@ | Out-Null

  # Clean up local artifacts so the next deploy regenerates them.
  $localConfig = Join-Path $ScriptRoot "config"
  foreach ($f in @("join-command.sh", "kubeconfig", "pending-labels.txt")) {
    $p = Join-Path $localConfig $f
    if (Test-Path $p) { Remove-Item $p -Force }
  }

  Write-Host "`n[OK] Cluster cleaned up on all nodes." -ForegroundColor Green
}

function Show-Status {
  param($Config)

  Write-Host "`n=== Cluster Status ===" -ForegroundColor Yellow

  $cpHost = $Config["CONTROL_PLANE_IP"]
  $cpUser = $Config["CONTROL_PLANE_USER"]

  Write-Host "`nNodes:" -ForegroundColor Cyan
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "kubectl get nodes -o wide 2>/dev/null || echo 'Cluster not reachable'" | Out-Null

  $ns = if ($config.ContainsKey('OSMO_NAMESPACE')) { $config['OSMO_NAMESPACE'] } else { 'osmo' }

  Write-Host "`nOSMO Pods:" -ForegroundColor Cyan
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "kubectl get pods -n $ns 2>/dev/null || echo 'No OSMO namespace found'" | Out-Null

  Write-Host "`nOSMO Services:" -ForegroundColor Cyan
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "kubectl get svc -n $ns 2>/dev/null || echo 'No OSMO namespace found'" | Out-Null
}

function Show-GpuStatus {
  param($Config)

  Write-Host "`n=== GPU Status Across Cluster ===" -ForegroundColor Yellow

  # Control plane GPU
  $cpHost = $Config["CONTROL_PLANE_IP"]
  $cpUser = $Config["CONTROL_PLANE_USER"]

  Write-Host "`nControl Plane ($cpHost):" -ForegroundColor Cyan
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,persistence_mode --format=csv,noheader 2>/dev/null || echo 'No GPU found'" | Out-Null

  # Worker GPUs
  $workerIps = $Config["WORKER_IPS"]
  $workerUsers = $Config["WORKER_USERS"]

  for ($i = 0; $i -lt $workerIps.Count; $i++) {
    $wHost = $workerIps[$i]
    $wUser = $workerUsers[$i]
    Write-Host "`nWorker $($i + 1) ($wHost):" -ForegroundColor Cyan
    Invoke-SshCommand -TargetHost $wHost -User $wUser -Command "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,persistence_mode --format=csv,noheader 2>/dev/null || echo 'No GPU found'" | Out-Null
  }

  # Kubernetes GPU allocation
  Write-Host "`nKubernetes GPU Allocation:" -ForegroundColor Cyan
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "kubectl describe nodes | grep -E 'Name:|nvidia.com/gpu' | head -20 2>/dev/null || echo 'Cannot query cluster'" | Out-Null

  # GPU Operator pods
  Write-Host "`nGPU Operator Pods:" -ForegroundColor Cyan
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "kubectl get pods -n gpu-operator --no-headers 2>/dev/null | awk '{print \$1, \$3}' || echo 'No GPU Operator namespace'" | Out-Null

  # Running GPU workloads
  Write-Host "`nActive GPU Workloads:" -ForegroundColor Cyan
  Invoke-SshCommand -TargetHost $cpHost -User $cpUser -Command "kubectl get pods --all-namespaces -o json 2>/dev/null | python3 -c ""
import json, sys
data = json.load(sys.stdin)
for pod in data.get('items', []):
  for c in pod['spec'].get('containers', []):
    res = c.get('resources', {})
    limits = res.get('limits', {})
    if 'nvidia.com/gpu' in limits:
      ns = pod['metadata']['namespace']
      name = pod['metadata']['name']
      gpu = limits['nvidia.com/gpu']
      phase = pod['status'].get('phase', 'Unknown')
      print(f'  {ns}/{name}: {gpu} GPU(s) [{phase}]')
"" 2>/dev/null || echo '  No GPU workloads found'" | Out-Null
}

#endregion

#region Main

function Main {
  Write-Host "============================================================" -ForegroundColor Green
  Write-Host "  NVIDIA OSMO Cluster Deployment Orchestrator" -ForegroundColor Green
  Write-Host "============================================================" -ForegroundColor Green
  Write-Host ""

  # Verify SSH is available
  if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Error "SSH client is required. Install OpenSSH or use Windows Terminal."
    return
  }

  $config = Read-InventoryConfig -Path $ConfigFile
  if (-not $config) { return }

  Write-Host "Configuration loaded:" -ForegroundColor Gray
  Write-Host "  Control Plane: $($config['CONTROL_PLANE_IP']) ($($config['CONTROL_PLANE_HOSTNAME']))" -ForegroundColor Gray
  Write-Host "  Workers:       $($config['WORKER_IPS'] -join ', ')" -ForegroundColor Gray
  Write-Host "  GPU Enabled:   $(if ($config.ContainsKey('ENABLE_GPU')) { $config['ENABLE_GPU'] } else { 'false' })" -ForegroundColor Gray
  Write-Host ""

  switch ($Action) {
    "all" {
      Install-Prerequisites -Config $config
      Initialize-ControlPlane -Config $config
      Join-Workers -Config $config
      Install-Osmo -Config $config
      Install-OsmoCli
      Show-Status -Config $config
    }
    "prerequisites" { Install-Prerequisites -Config $config }
    "control-plane" { Initialize-ControlPlane -Config $config }
    "workers"       { Join-Workers -Config $config }
    "osmo"          { Install-Osmo -Config $config }
    "cli"           { Install-OsmoCli }
    "cleanup"       { Invoke-Cleanup -Config $config }
    "status"        { Show-Status -Config $config }
    "gpu-status"    { Show-GpuStatus -Config $config }
    "local"         { Install-LocalOsmo }
    "add-node"      { Add-Node -Config $config }
  }

  Write-Host "`n============================================================" -ForegroundColor Green
  Write-Host "  Deployment action '$Action' complete." -ForegroundColor Green
  Write-Host "============================================================" -ForegroundColor Green
}

Main

#endregion
