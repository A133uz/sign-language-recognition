$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try {
    $taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run scripts/setup.ps1 first.' }
    $taskUvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    if ($taskUvCommand) {
        $taskUv = $taskUvCommand.Source
    }
    else {
        $taskUv = Join-Path $env:USERPROFILE '.local/bin/uv.exe'
        if (-not (Test-Path -LiteralPath $taskUv)) {
            throw 'Install standalone uv 0.13.5 first; see docs/setup.md.'
        }
    }
    & $taskUv --cache-dir .cache/uv lock --check --offline
    if ($LASTEXITCODE -ne 0) { throw 'Dependency lock is out of date.' }
    & $taskPython -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw 'Lint failed.' }
    & $taskPython -m ruff format --check .
    if ($LASTEXITCODE -ne 0) { throw 'Formatting failed.' }
    & $taskPython -m pytest
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
    foreach ($taskVocabulary in Get-ChildItem -LiteralPath configs -Filter 'vocabulary*.yaml' -File) {
        & $taskPython -m sign_language check-config $taskVocabulary.FullName
        if ($LASTEXITCODE -ne 0) { throw "Vocabulary validation failed: $($taskVocabulary.Name)" }
    }
    & $taskPython -m sign_language doctor
    if ($LASTEXITCODE -ne 0) { throw 'Base environment checks failed.' }
    $taskGitRoot = $taskRoot.Replace('\', '/')
    git -c "safe.directory=$taskGitRoot" diff --check
    if ($LASTEXITCODE -ne 0) { throw 'Git whitespace checks failed.' }
}
finally {
    Pop-Location
}
