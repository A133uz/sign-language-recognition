$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $taskRoot '.venv/Scripts/Activate.ps1')
$taskUvBin = Join-Path $taskRoot '.cache/uv/bin'
if (Test-Path (Join-Path $taskUvBin 'uv.exe')) {
    $env:PATH = "$taskUvBin;$env:PATH"
}
Write-Output 'Activated the project .venv.'
