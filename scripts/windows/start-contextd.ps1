# Bootstrap script for Task Scheduler — waits for Docker, then runs contextd up.
# Logs to ~/.contextd/logs/startup.log for debugging.

$logDir = "$env:USERPROFILE\.contextd\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$log = "$logDir\startup.log"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts  $msg" | Out-File -Append -FilePath $log
}

Log "contextd startup initiated"

# Wait up to 120s for Docker to be responsive.
$deadline = (Get-Date).AddSeconds(120)
$dockerReady = $false
while ((Get-Date) -lt $deadline) {
    $out = docker info 2>&1
    if ($LASTEXITCODE -eq 0) { $dockerReady = $true; break }
    Start-Sleep -Seconds 5
}

if (-not $dockerReady) {
    Log "ABORT: Docker not responsive after 120s"
    exit 1
}

Log "Docker ready"

# Activate venv and run contextd up.
$venvActivate = "C:\Users\giuse\src\contextd\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Log "ABORT: venv not found at $venvActivate"
    exit 1
}

& $venvActivate
contextd up 2>&1 | Out-File -Append -FilePath $log
Log "contextd up exited with code $LASTEXITCODE"
