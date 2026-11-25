# Supervisor cho 1 thành phần của Magician: chạy process, chết thì 5s sau chạy lại. Task Scheduler gọi script này.
#   powershell -File scripts\run_service.ps1 -Name api      # uvicorn
#   powershell -File scripts\run_service.ps1 -Name tunnel   # cloudflared tunnel magician
param([Parameter(Mandatory)][ValidateSet("api", "tunnel")][string]$Name)

$ErrorActionPreference = "Continue"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = if ($env:MAGICIAN_PYTHON) { $env:MAGICIAN_PYTHON } else { "D:\envs\omnivoice\python.exe" }
$Cloudflared = if ($env:MAGICIAN_CLOUDFLARED) { $env:MAGICIAN_CLOUDFLARED } else { "C:\Program Files (x86)\cloudflared\cloudflared.exe" }
$TunnelConfig = if ($env:MAGICIAN_TUNNEL_CONFIG) { $env:MAGICIAN_TUNNEL_CONFIG } else { Join-Path $env:USERPROFILE ".cloudflared\magician.yml" }
$LogDir = Join-Path $Root "data\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Cache model / temp trên ổ D (ổ C gần đầy => lỗi paging file)
if (-not $env:HF_HOME) { $env:HF_HOME = "D:\cache\hf" }
if (-not $env:MAGICIAN_TEMP) { $env:MAGICIAN_TEMP = "D:\cache\tmp" }
$env:TEMP = $env:MAGICIAN_TEMP
$env:TMP = $env:MAGICIAN_TEMP
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

if ($Name -eq "api") {
    $exe = $Python
    $argv = @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--proxy-headers")
} else {
    $exe = $Cloudflared
    $argv = @("--config", $TunnelConfig, "--no-autoupdate", "tunnel", "run")
}
$out = Join-Path $LogDir "$Name.out.log"
$err = Join-Path $LogDir "$Name.err.log"

function Write-Sup($msg) {
    Add-Content -Path (Join-Path $LogDir "supervisor.log") -Value "$(Get-Date -Format s) [$Name] $msg" -Encoding utf8
}

while ($true) {
    # giữ log stdout/stderr không phình: > 50MB thì xoay sang .1
    foreach ($f in @($out, $err)) {
        if ((Test-Path $f) -and (Get-Item $f).Length -gt 50MB) { Move-Item -Force $f "$f.1" }
    }
    Write-Sup "start: $exe $($argv -join ' ')"
    $p = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $Root -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $p.WaitForExit()
    Write-Sup "exited with code $($p.ExitCode), restart in 5s"
    Start-Sleep -Seconds 5
}
