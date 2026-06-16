<#
.SYNOPSIS
  Quick helper script for common OSMO deployment tasks.

.DESCRIPTION
  Wraps deploy.ps1 with simpler commands and adds pre-flight checks.
  Use this when you don't want to remember deploy.ps1 parameters.

.EXAMPLE
  .\osmo.ps1 check        # Verify prerequisites and connectivity
  .\osmo.ps1 deploy       # Full deployment from scratch
  .\osmo.ps1 status       # Show cluster + OSMO status
  .\osmo.ps1 gpu          # Show GPU status on all nodes
  .\osmo.ps1 redeploy     # Tear down and redeploy OSMO only
  .\osmo.ps1 teardown     # Full cluster teardown
  .\osmo.ps1 add 10.0.0.5 myuser  # Add a new worker node
  .\osmo.ps1 ssh cp       # SSH into the control plane
  .\osmo.ps1 ssh 1        # SSH into worker 1
  .\osmo.ps1 ui           # Open OSMO UI in default browser
  .\osmo.ps1 version      # Show deployed OSMO version
#>

param(
  [Parameter(Position = 0)]
  [string]$Command = "help",

  [Parameter(Position = 1)]
  [string]$Arg1 = "",

  [Parameter(Position = 2)]
  [string]$Arg2 = ""
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

#region Config Loading

function Get-Config {
  $configPath = Join-Path $ScriptRoot "config\inventory.env"
  if (-not (Test-Path $configPath)) {
    Write-Error "Config not found: $configPath`nRun: cp config\inventory.example.env config\inventory.env"
    return $null
  }

  $config = @{}
  $content = Get-Content $configPath -Raw

  foreach ($line in ($content -split "`n")) {
    $line = $line.Trim()
    if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
    if ($line -match '^\s*([A-Z_]+)\s*=\s*"([^"]*)"') {
      $config[$Matches[1]] = $Matches[2]
    }
  }

  # Parse arrays
  if ($content -match 'WORKER_IPS=\(([^)]+)\)') {
    $config["WORKER_IPS"] = ($Matches[1] -replace '"', '').Trim() -split '\s+'
  }
  if ($content -match 'WORKER_USERS=\(([^)]+)\)') {
    $config["WORKER_USERS"] = ($Matches[1] -replace '"', '').Trim() -split '\s+'
  }
  if ($content -match 'WORKER_HOSTNAMES=\(([^)]+)\)') {
    $config["WORKER_HOSTNAMES"] = ($Matches[1] -replace '"', '').Trim() -split '\s+'
  }

  return $config
}

function Get-SshKey {
  $key = "$env:USERPROFILE\.ssh\id_ed25519"
  if (-not (Test-Path $key)) {
    $key = "$env:USERPROFILE\.ssh\id_rsa"
  }
  return $key
}

function Invoke-NodeSsh {
  param([string]$HostName, [string]$User, [string]$Cmd)
  $key = Get-SshKey
  ssh -i $key -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${User}@${HostName}" $Cmd
}

#endregion

#region Commands

function Show-Help {
  Write-Host ""
  Write-Host "  NVIDIA OSMO Deployment Helper" -ForegroundColor Green
  Write-Host "  =============================" -ForegroundColor Green
  Write-Host ""
  Write-Host "  Usage: .\osmo.ps1 <command> [args]" -ForegroundColor White
  Write-Host ""
  Write-Host "  Commands:" -ForegroundColor Yellow
  Write-Host "    check         Pre-flight checks (SSH, connectivity, config)" -ForegroundColor White
  Write-Host "    deploy        Full deployment from scratch" -ForegroundColor White
  Write-Host "    status        Show cluster and OSMO pod status" -ForegroundColor White
  Write-Host "    gpu           Show GPU health on all nodes" -ForegroundColor White
  Write-Host "    version       Show deployed OSMO version" -ForegroundColor White
  Write-Host "    redeploy      Tear down OSMO and redeploy (keeps K8s)" -ForegroundColor White
  Write-Host "    teardown      Full cluster teardown" -ForegroundColor White
  Write-Host "    add <ip> [user]  Add a new worker node" -ForegroundColor White
  Write-Host "    ssh cp|<n>    SSH into control plane or worker N" -ForegroundColor White
  Write-Host "    ui            Open OSMO UI in browser" -ForegroundColor White
  Write-Host "    logs          Show OSMO service logs" -ForegroundColor White
  Write-Host "    help          Show this help" -ForegroundColor White
  Write-Host ""
  Write-Host "  Examples:" -ForegroundColor Yellow
  Write-Host "    .\osmo.ps1 check" -ForegroundColor Gray
  Write-Host "    .\osmo.ps1 deploy" -ForegroundColor Gray
  Write-Host "    .\osmo.ps1 add 192.168.1.104 edge" -ForegroundColor Gray
  Write-Host "    .\osmo.ps1 ssh cp" -ForegroundColor Gray
  Write-Host ""
}

function Test-Preflight {
  $config = Get-Config
  if (-not $config) { return }
  $key = Get-SshKey
  $allOk = $true

  Write-Host "`n  Pre-Flight Checks" -ForegroundColor Yellow
  Write-Host "  =================" -ForegroundColor Yellow

  # Check SSH client
  Write-Host -NoNewline "  [1] SSH client:          "
  if (Get-Command ssh -ErrorAction SilentlyContinue) {
    Write-Host "OK" -ForegroundColor Green
  } else {
    Write-Host "MISSING" -ForegroundColor Red; $allOk = $false
  }

  # Check SSH key
  Write-Host -NoNewline "  [2] SSH key ($key): "
  if (Test-Path $key) {
    Write-Host "OK" -ForegroundColor Green
  } else {
    Write-Host "NOT FOUND" -ForegroundColor Red; $allOk = $false
  }

  # Check control plane connectivity
  $cpIp = $config["CONTROL_PLANE_IP"]
  $cpUser = $config["CONTROL_PLANE_USER"]
  Write-Host -NoNewline "  [3] Control plane ($cpIp): "
  ping -n 1 -w 3000 $cpIp 2>&1 | Out-Null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "REACHABLE" -ForegroundColor Green
  } else {
    Write-Host "UNREACHABLE" -ForegroundColor Red; $allOk = $false
  }

  # Check SSH to control plane
  Write-Host -NoNewline "  [4] SSH to control plane:    "
  try {
    $result = ssh -i $key -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes "${cpUser}@${cpIp}" "echo OK" 2>&1
    if ($result -match "OK") {
      Write-Host "OK" -ForegroundColor Green
    } else {
      Write-Host "FAILED" -ForegroundColor Red; $allOk = $false
    }
  } catch {
    Write-Host "FAILED" -ForegroundColor Red; $allOk = $false
  }

  # Check workers
  $workerIps = $config["WORKER_IPS"]
  $workerUsers = $config["WORKER_USERS"]
  for ($i = 0; $i -lt $workerIps.Count; $i++) {
    $wIp = $workerIps[$i]
    Write-Host -NoNewline "  [$($i+5)] Worker $($i+1) ($wIp):      "
    ping -n 1 -w 3000 $wIp 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
      Write-Host "REACHABLE" -ForegroundColor Green
    } else {
      Write-Host "UNREACHABLE" -ForegroundColor Red; $allOk = $false
    }
  }

  # Config summary
  Write-Host ""
  Write-Host "  Configuration:" -ForegroundColor Cyan
  Write-Host "    OSMO version:    $($config['OSMO_IMAGE_TAG'])" -ForegroundColor Gray
  Write-Host "    GPU enabled:     $($config['ENABLE_GPU'])" -ForegroundColor Gray
  Write-Host "    Untaint CP:      $($config['UNTAINT_CONTROL_PLANE'])" -ForegroundColor Gray
  Write-Host "    Nodes:           1 CP + $($workerIps.Count) workers = $($workerIps.Count + 1) total" -ForegroundColor Gray
  Write-Host ""

  if ($allOk) {
    Write-Host "  All checks passed. Ready to deploy!" -ForegroundColor Green
  } else {
    Write-Host "  Some checks failed. Fix issues above before deploying." -ForegroundColor Red
  }
  Write-Host ""
}

function Get-OsmoVersion {
  $config = Get-Config
  if (-not $config) { return }

  $cpIp = $config["CONTROL_PLANE_IP"]
  $cpUser = $config["CONTROL_PLANE_USER"]

  Write-Host "`n  OSMO Version Info" -ForegroundColor Yellow
  Write-Host "  =================" -ForegroundColor Yellow

  $helmOutput = Invoke-NodeSsh -HostName $cpIp -User $cpUser -Cmd "helm list -n osmo --output json 2>/dev/null"
  if ($helmOutput) {
    try {
      $releases = $helmOutput | ConvertFrom-Json
      foreach ($r in $releases) {
        Write-Host "    Chart:       $($r.chart)" -ForegroundColor White
        Write-Host "    App Version: $($r.app_version)" -ForegroundColor White
        Write-Host "    Status:      $($r.status)" -ForegroundColor White
        Write-Host "    Deployed:    $($r.updated)" -ForegroundColor White
      }
    } catch {
      Write-Host "    Could not parse Helm output." -ForegroundColor Red
    }
  } else {
    Write-Host "    OSMO is not installed." -ForegroundColor Yellow
  }

  Write-Host ""
  Write-Host "    Config tag:  $($config['OSMO_IMAGE_TAG'])" -ForegroundColor Gray
  Write-Host ""
}

function Open-SshSession {
  param([string]$Target)
  $config = Get-Config
  if (-not $config) { return }
  $key = Get-SshKey

  if ($Target -eq "cp" -or $Target -eq "0" -or $Target -eq "control-plane") {
    $ip = $config["CONTROL_PLANE_IP"]
    $user = $config["CONTROL_PLANE_USER"]
  } else {
    $idx = [int]$Target - 1
    $workerIps = $config["WORKER_IPS"]
    $workerUsers = $config["WORKER_USERS"]
    if ($idx -lt 0 -or $idx -ge $workerIps.Count) {
      Write-Error "Invalid worker index '$Target'. Valid: cp, 1-$($workerIps.Count)"
      return
    }
    $ip = $workerIps[$idx]
    $user = $workerUsers[$idx]
  }

  Write-Host "Connecting to ${user}@${ip}..." -ForegroundColor Cyan
  ssh -i $key -o StrictHostKeyChecking=no "${user}@${ip}"
}

function Open-OsmoUI {
  $config = Get-Config
  if (-not $config) { return }
  $hostname = if ($config.ContainsKey('OSMO_HOSTNAME')) { $config['OSMO_HOSTNAME'] } else { 'quick-start.osmo' }
  $url = "http://${hostname}:30080"
  Write-Host "Opening $url ..." -ForegroundColor Cyan
  Write-Host "(Make sure $($config['CONTROL_PLANE_IP'])  $hostname is in your hosts file)" -ForegroundColor Yellow
  Start-Process $url
}

function Show-OsmoLogs {
  $config = Get-Config
  if (-not $config) { return }
  $cpIp = $config["CONTROL_PLANE_IP"]
  $cpUser = $config["CONTROL_PLANE_USER"]
  $ns = if ($config.ContainsKey('OSMO_NAMESPACE')) { $config['OSMO_NAMESPACE'] } else { 'osmo' }

  Write-Host "`n  Recent OSMO Service Logs" -ForegroundColor Yellow
  $output = Invoke-NodeSsh -HostName $cpIp -User $cpUser -Cmd "kubectl logs -n $ns -l app=osmo-service --tail=50 2>/dev/null || echo 'No logs available'"
  $output | ForEach-Object { Write-Host "  $_" }
}

#endregion

#region Main

$resolvedKey = Get-SshKey

switch ($Command.ToLower()) {
  "help"      { Show-Help }
  "check"     { Test-Preflight }
  "deploy"    { & "$ScriptRoot\deploy.ps1" -Action all -SshKeyPath $resolvedKey }
  "status"    { & "$ScriptRoot\deploy.ps1" -Action status -SshKeyPath $resolvedKey }
  "gpu"       { & "$ScriptRoot\deploy.ps1" -Action gpu-status -SshKeyPath $resolvedKey }
  "version"   { Get-OsmoVersion }
  "redeploy"  {
    Write-Host "This will remove OSMO and redeploy it (Kubernetes stays intact)." -ForegroundColor Yellow
    $confirm = Read-Host "Continue? (y/N)"
    if ($confirm -eq 'y') {
      $config = Get-Config
      $cpIp = $config["CONTROL_PLANE_IP"]
      $cpUser = $config["CONTROL_PLANE_USER"]
      Invoke-NodeSsh -HostName $cpIp -User $cpUser -Cmd "sudo bash ~/osmo-deployment/scripts/05-cleanup.sh --keep-k8s --config ~/osmo-deployment/config/inventory.env"
      & "$ScriptRoot\deploy.ps1" -Action osmo -SshKeyPath $resolvedKey
    }
  }
  "teardown"  {
    Write-Host "WARNING: This will destroy the entire cluster and remove Kubernetes." -ForegroundColor Red
    $confirm = Read-Host "Type 'yes' to confirm"
    if ($confirm -eq 'yes') {
      & "$ScriptRoot\deploy.ps1" -Action cleanup -SshKeyPath $resolvedKey
    }
  }
  "add"       {
    if (-not $Arg1) {
      Write-Error "Usage: .\osmo.ps1 add <ip> [user]"
      return
    }
    $nodeUser = if ($Arg2) { $Arg2 } else { "ubuntu" }
    & "$ScriptRoot\deploy.ps1" -Action add-node -NodeIp $Arg1 -NodeUser $nodeUser -SshKeyPath $resolvedKey
  }
  "ssh"       { Open-SshSession -Target $(if ($Arg1) { $Arg1 } else { "cp" }) }
  "ui"        { Open-OsmoUI }
  "logs"      { Show-OsmoLogs }
  default     { Write-Error "Unknown command: $Command. Run '.\osmo.ps1 help' for usage." }
}

#endregion
