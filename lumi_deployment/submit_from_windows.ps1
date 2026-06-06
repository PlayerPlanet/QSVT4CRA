# =============================================================================
# submit_from_windows.ps1
# -----------------------------------------------------------------------------
# One-command submit of QSVT4CRA experiments to LUMI from a Windows host.
#
# Usage (PowerShell):
#   cd C:\Users\Käyttäjä\Documents\projects\QSVT4CRA
#   .\lumi_deployment\submit_from_windows.ps1                     # smoke + everything
#   .\lumi_deployment\submit_from_windows.ps1 -SmokeOnly          # smoke test only
#   .\lumi_deployment\submit_from_windows.ps1 -SkipGpu -SkipWeak  # CPU strong only
#   .\lumi_deployment\submit_from_windows.ps1 -DryRun             # show what would be submitted
#
# First-time setup (one-time):
#   1. OpenSSH client is built into Windows 10 1809+; usually no install needed
#   2. Set up an SSH config alias `lumi` so `ssh lumi` works:
#        notepad $HOME\.ssh\config
#      Add:
#        Host lumi
#          HostName lumi.csc.fi
#          User kkiirikk
#          ServerAliveInterval 60
#          ServerAliveCountMax 3
#   3. (One-time) Add your CSC/MyAccessID SSH key to the ssh-agent
#        ssh-add $HOME\.ssh\lumi_key
#
# This script:
#   1. rsyncs the project to LUMI scratch (reuses rsync_to_lumi.sh)
#   2. installs Python deps on LUMI (one-time, idempotent)
#   3. dispatches the experiment matrix (reuses dispatch_all.sh)
# =============================================================================

[CmdletBinding()]
param(
    [switch]$SmokeOnly = $false,
    [switch]$SkipGpu = $false,
    [switch]$SkipWeak = $false,
    [switch]$SkipHybrid = $false,
    [switch]$SkipStrong = $false,
    [switch]$DryRun = $false,
    [string]$LumiUser = "kkiirikk",
    [string]$LumiHost = "lumi.csc.fi",
    [string]$Account = "project_465003017",
    [string]$Scratch = "/scratch/project_465003017/kkiirikk/qsvt4cra-research",
    [int]$MaxParallel = 4,
    [switch]$SkipRsync = $false,
    [switch]$SkipInstall = $false,
    [switch]$Help = $false
)

$ErrorActionPreference = "Stop"

# ----------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------
if ($Help) {
    Get-Help $PSCommandPath -Full
    exit 0
}

# ----------------------------------------------------------------------
# Environment detection
# ----------------------------------------------------------------------
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Remote = "${LumiUser}@${LumiHost}"
$RemoteScratch = "/scratch/${Account}/${LumiUser}/qsvt4cra-research"

Write-Host "============================================================"
Write-Host "QSVT4CRA submit_from_windows.ps1"
Write-Host "============================================================"
Write-Host "  Project root    : $ProjectRoot"
Write-Host "  LUMI user       : $LumiUser"
Write-Host "  LUMI host       : $LumiHost"
Write-Host "  Scratch (local) : $Scratch"
Write-Host "  Scratch (remote): $RemoteScratch"
Write-Host "  Smoke only      : $SmokeOnly"
Write-Host "  Skip-GPU        : $SkipGpu"
Write-Host "  Skip-Weak       : $SkipWeak"
Write-Host "  Skip-Hybrid     : $SkipHybrid"
Write-Host "  Skip-Strong     : $SkipStrong"
Write-Host "  Dry run         : $DryRun"
Write-Host "  Skip rsync      : $SkipRsync"
Write-Host "  Skip install    : $SkipInstall"
Write-Host ""

# ----------------------------------------------------------------------
# Pre-flight: verify SSH connectivity
# ----------------------------------------------------------------------
Write-Host "[1/5] Testing SSH connectivity..."
$whoami = ssh "$Remote" "whoami" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "SSH to $Remote failed: $whoami"
    exit 1
}
Write-Host "  -> $whoami"

# ----------------------------------------------------------------------
# Step 2: rsync project to LUMI scratch
# ----------------------------------------------------------------------
if ($SkipRsync) {
    Write-Host "[2/5] Skipping rsync (SkipRsync=true)"
} else {
    Write-Host "[2/5] rsync to LUMI scratch..."
    $rsyncScript = Join-Path $PSScriptRoot "rsync_to_lumi.sh"
    if (-not (Test-Path $rsyncScript)) {
        Write-Error "rsync_to_lumi.sh not found: $rsyncScript"
        exit 1
    }
    # Run the rsync script via bash
    $env:REMOTE = $Remote
    $env:ACCOUNT = $Account
    bash $rsyncScript
    if ($LASTEXITCODE -ne 0) {
        Write-Error "rsync failed with exit $LASTEXITCODE"
        exit 1
    }
}

# ----------------------------------------------------------------------
# Step 3: install Python deps (idempotent)
# ----------------------------------------------------------------------
if ($SkipInstall) {
    Write-Host "[3/5] Skipping pip install (SkipInstall=true)"
} else {
    Write-Host "[3/5] Installing Python dependencies on LUMI..."
    ssh $Remote "cd $RemoteScratch && bash -lc 'source lumi_deployment/setup_lumi_env.sh && if [ ! -d site-packages/torch ] || [ -z \"\$(ls site-packages/torch 2>/dev/null)\" ]; then echo \"Installing...\"; pip install --no-cache-dir --target=./site-packages -r requirements.txt 2>&1 | tail -20; else echo \"Already installed (site-packages/torch exists)\"; fi'"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "pip install may have failed; check logs"
    }
}

# ----------------------------------------------------------------------
# Step 4: build dispatch args and run dispatch_all.sh
# ----------------------------------------------------------------------
Write-Host "[4/5] Building dispatch arguments..."
$dispatchArgs = @()
if ($SmokeOnly)   { $dispatchArgs += "--smoke-only" }
if ($SkipGpu)     { $dispatchArgs += "--skip-gpu" }
if ($SkipWeak)    { $dispatchArgs += "--skip-weak" }
if ($SkipHybrid)  { $dispatchArgs += "--skip-hybrid" }
if ($SkipStrong)  { $dispatchArgs += "--skip-strong" }
if ($DryRun)      { $dispatchArgs += "--dry-run" }
$dispatchArgs += "--max-parallel=$MaxParallel"

Write-Host "  dispatch args: $($dispatchArgs -join ' ')"

# ----------------------------------------------------------------------
# Step 5: dispatch the experiment matrix
# ----------------------------------------------------------------------
Write-Host "[5/5] Dispatching experiment matrix..."
$dispatchScript = "lumi_deployment/dispatch_all.sh"
$remoteCmd = "cd $RemoteScratch && bash $dispatchScript $($dispatchArgs -join ' ')"
Write-Host "  Remote command: $remoteCmd"
Write-Host ""

ssh $Remote $remoteCmd
$dispatchExit = $LASTEXITCODE

Write-Host ""
if ($dispatchExit -eq 0) {
    Write-Host "============================================================"
    Write-Host "Dispatch complete!"
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "Next: monitor jobs with:"
    Write-Host "  ssh $Remote 'squeue -u $LumiUser'"
    Write-Host "  ssh $Remote 'sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem'"
    Write-Host ""
    Write-Host "Tail logs:"
    Write-Host "  ssh $Remote 'ls -lt $RemoteScratch/slurm_logs/ | head'"
    Write-Host ""
} else {
    Write-Error "Dispatch failed with exit $dispatchExit"
    exit $dispatchExit
}
