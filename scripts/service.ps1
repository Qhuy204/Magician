# Quản lý Magician bằng Windows Task Scheduler (2 task: Magician-API, Magician-Tunnel).
#   powershell -ExecutionPolicy Bypass -File scripts\service.ps1 install    # tạo task, chạy khi đăng nhập + tự restart
#   powershell -ExecutionPolicy Bypass -File scripts\service.ps1 start|stop|restart|status|logs|uninstall
param([Parameter(Mandatory)][ValidateSet("install", "uninstall", "start", "stop", "restart", "status", "logs")][string]$Action,
      [ValidateSet("all", "api", "tunnel")][string]$Only = "all")

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Runner = Join-Path $PSScriptRoot "run_service.ps1"
$Tasks = [ordered]@{ api = "Magician-API"; tunnel = "Magician-Tunnel" }
$Selected = if ($Only -eq "all") { @($Tasks.Keys) } else { @($Only) }
# Dấu hiệu nhận ra process con khi dừng (Stop-ScheduledTask chỉ dừng supervisor powershell)
$Patterns = @{ api = @("*uvicorn app.main:app*", "*omnivoice_worker.py*", "*xtts_worker.py*", "*run_service.ps1 -Name api*");
               tunnel = @("*magician.yml*", "*run_service.ps1 -Name tunnel*") }

function Kill-Children($key) {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $cl = $_.CommandLine; $cl -and ($Patterns[$key] | Where-Object { $cl -like $_ })
    }
    foreach ($p in $procs) {
        if ($p.ProcessId -ne $PID) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}

function Install-One($key) {
    $name = $Tasks[$key]
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $Root `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Runner`" -Name $key"
    $triggers = @(New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME")
    # Supervisor tự restart process con; task tự chạy lại supervisor nếu chính nó chết
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Settings $settings -Principal $principal `
        -Description "Magician $key (supervisor: scripts\run_service.ps1)" -Force | Out-Null
    Write-Host "installed $name"
}

switch ($Action) {
    "install" { foreach ($k in $Selected) { Install-One $k; Start-ScheduledTask -TaskName $Tasks[$k] } }
    "uninstall" {
        foreach ($k in $Selected) {
            Stop-ScheduledTask -TaskName $Tasks[$k] -ErrorAction SilentlyContinue
            Kill-Children $k
            Unregister-ScheduledTask -TaskName $Tasks[$k] -Confirm:$false -ErrorAction SilentlyContinue
            Write-Host "removed $($Tasks[$k])"
        }
    }
    "start" { foreach ($k in $Selected) { Start-ScheduledTask -TaskName $Tasks[$k]; Write-Host "started $($Tasks[$k])" } }
    "stop" {
        foreach ($k in $Selected) {
            Stop-ScheduledTask -TaskName $Tasks[$k] -ErrorAction SilentlyContinue
            Kill-Children $k
            Write-Host "stopped $($Tasks[$k])"
        }
    }
    "restart" {
        foreach ($k in $Selected) {
            Stop-ScheduledTask -TaskName $Tasks[$k] -ErrorAction SilentlyContinue
            Kill-Children $k
            Start-Sleep -Seconds 2
            Start-ScheduledTask -TaskName $Tasks[$k]
            Write-Host "restarted $($Tasks[$k])"
        }
    }
    "status" {
        foreach ($k in $Selected) {
            $t = Get-ScheduledTask -TaskName $Tasks[$k] -ErrorAction SilentlyContinue
            $info = if ($t) { Get-ScheduledTaskInfo -TaskName $Tasks[$k] } else { $null }
            "{0,-16} {1,-10} last run {2} (result {3})" -f $Tasks[$k], ($(if ($t) { $t.State } else { "missing" })),
                $info.LastRunTime, $info.LastTaskResult
        }
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:8000/ready"
            "local /ready: $($r.StatusCode) $($r.Content)"
        } catch { "local /ready: $($_.Exception.Message)" }
    }
    "logs" {
        $f = Join-Path $Root "data\logs\magician.log"
        Get-Content $f -Tail 50
        Write-Host "---- supervisor ----"
        Get-Content (Join-Path $Root "data\logs\supervisor.log") -Tail 10 -ErrorAction SilentlyContinue
    }
}
