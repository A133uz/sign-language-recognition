param(
    [ValidateSet('dev', 'cpu', 'cuda')]
    [string]$Profile = 'cuda',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try {
    $taskPythonExecutable = & $Python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'; print(sys.executable)"
    if ($LASTEXITCODE -ne 0) { throw 'Select a Python 3.12 interpreter with -Python.' }

    $taskUvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    if ($taskUvCommand) {
        $taskUv = $taskUvCommand.Source
    }
    else {
        # An already-open terminal may not have the installer's updated PATH yet.
        $taskUv = Join-Path $env:USERPROFILE '.local/bin/uv.exe'
        if (-not (Test-Path -LiteralPath $taskUv)) {
            throw 'Install standalone uv 0.13.5 first; see docs/setup.md.'
        }
    }
    $taskArgs = @('--cache-dir', '.cache/uv', 'sync', '--locked', '--python', $taskPythonExecutable)
    if ($Profile -ne 'dev') {
        $taskArgs += @('--extra', 'vision', '--extra', 'training', '--extra', $Profile)
    }
    & $taskUv @taskArgs
    if ($LASTEXITCODE -ne 0) { throw 'Dependency synchronization failed.' }

    $taskProjectPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
    & $taskProjectPython -m sign_language check-config configs/vocabulary.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Vocabulary validation failed.' }
    $taskDoctorArgs = @('-m', 'sign_language', 'doctor')
    if ($Profile -ne 'dev') { $taskDoctorArgs += '--vision' }
    if ($Profile -eq 'cpu') { $taskDoctorArgs += '--torch' }
    if ($Profile -eq 'cuda') { $taskDoctorArgs += '--require-cuda' }
    & $taskProjectPython @taskDoctorArgs
    if ($LASTEXITCODE -ne 0) { throw 'Environment checks failed.' }
    Write-Host "Setup complete: $Profile profile. Run .\scripts\check.ps1 to check the repository."
}
finally {
    Pop-Location
}
