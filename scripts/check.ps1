$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try {
    $taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run scripts/setup.ps1 first.' }
    $taskUv = Join-Path $taskRoot '.bootstrap/Scripts/uv.exe'
    if (Test-Path -LiteralPath $taskUv) {
        & $taskUv --cache-dir .cache/uv lock --check --offline
        if ($LASTEXITCODE -ne 0) { throw 'Dependency lock is out of date.' }
    }
    & $taskPython -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw 'Lint failed.' }
    & $taskPython -m ruff format --check .
    if ($LASTEXITCODE -ne 0) { throw 'Formatting failed.' }
    & $taskPython -m pytest
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
    & $taskPython -m sign_language check-config configs/vocabulary.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Vocabulary validation failed.' }
    & $taskPython -m sign_language doctor
    if ($LASTEXITCODE -ne 0) { throw 'Base environment checks failed.' }
    $taskGitRoot = $taskRoot.Replace('\', '/')
    git -c "safe.directory=$taskGitRoot" diff --check
    if ($LASTEXITCODE -ne 0) { throw 'Git whitespace checks failed.' }
}
finally {
    Pop-Location
}
