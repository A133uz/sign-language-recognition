$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $taskRoot '.venv/Scripts/Activate.ps1')
Write-Output 'Activated the project .venv.'
