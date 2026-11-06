$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try {
    uv --cache-dir .cache/uv lock --check --offline
    if ($LASTEXITCODE -ne 0) { throw 'Lockfile check failed.' }
    & ./.venv/Scripts/python.exe -m compileall -q src scripts tests
    if ($LASTEXITCODE -ne 0) { throw 'Source compilation failed.' }
    $taskPreviousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $taskRoot 'src'
        & ./.venv/Scripts/python.exe -m unittest discover -s tests -v
        if ($LASTEXITCODE -ne 0) { throw 'Pipeline tests failed.' }
    }
    finally { $env:PYTHONPATH = $taskPreviousPythonPath }
    & ./.venv/Scripts/python.exe -c "import mediapipe,cv2,numpy; print('mediapipe',mediapipe.__version__,'opencv',cv2.__version__,'numpy',numpy.__version__); print('Import and unit checks do not validate live hand tracking or sign recognition.')"
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
}
finally { Pop-Location }
