param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$modelsRoot=[IO.Path]::GetFullPath((Join-Path $projectRoot 'models'))
$planData=Get-Content -LiteralPath $Plan -Raw | ConvertFrom-Json
if ($planData.status -ne 'planned' -or $planData.schema -ne 'model-retention-plan-v1') { throw 'Unexpected cleanup plan' }
if ([IO.Path]::GetFullPath($planData.root) -ne $projectRoot) { throw 'Plan belongs to another workspace' }
$historyRoot=[IO.Path]::GetFullPath((Join-Path $projectRoot $planData.history_root))
if (-not $historyRoot.StartsWith(($projectRoot+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)) { throw 'History outside workspace' }
if (Test-Path -LiteralPath $historyRoot) { throw 'Preserve existing cleanup history; use a new plan/version' }
function Get-SafeModelFile([string]$relative) {
    $absolute=[IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $absolute.StartsWith(($modelsRoot+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)) { throw "Target outside models: $absolute" }
    $resolved=(Resolve-Path -LiteralPath $absolute).ProviderPath
    if ($resolved -ne $absolute) { throw "Unexpected resolved target: $absolute" }
    return $absolute
}
# Check every final target and reject junctions/symlinks before any mutation.
if (((Get-Item -LiteralPath $modelsRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Model root is a reparse point' }
$reparse=@(Get-ChildItem -LiteralPath $modelsRoot -Recurse -Force | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 })
if ($reparse.Count) { throw 'Refusing cleanup through model symlinks/junctions' }
foreach ($entry in $planData.files) {
    $source=Get-SafeModelFile $entry.path
    if ((Get-Item -LiteralPath $source).Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Source changed since plan: $source" }
}
New-Item -ItemType Directory -Path $historyRoot | Out-Null
# Preserve and verify text/source/metric history only. No rejected weights are copied.
foreach ($entry in $planData.files | Where-Object action -eq 'preserve_metadata_then_remove') {
    $source=Get-SafeModelFile $entry.path
    $destination=[IO.Path]::GetFullPath((Join-Path $historyRoot $entry.path))
    if (-not $destination.StartsWith(($historyRoot+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe history destination' }
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'Metadata copy verification failed' }
}
$removed=0L
foreach ($entry in $planData.files | Where-Object action -ne 'keep') {
    $source=Get-SafeModelFile $entry.path
    Remove-Item -LiteralPath $source -Force
    $removed+=$entry.bytes
}
# Remove empty directories only, deepest first; never recursively delete a computed directory.
$directories=@(Get-ChildItem -LiteralPath $modelsRoot -Directory -Recurse -Force | Sort-Object { $_.FullName.Length } -Descending)
foreach ($directory in $directories) {
    $absolute=[IO.Path]::GetFullPath($directory.FullName)
    if (-not $absolute.StartsWith(($modelsRoot+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)) { throw 'Empty directory outside models' }
    if (@(Get-ChildItem -LiteralPath $absolute -Force).Count -eq 0) { Remove-Item -LiteralPath $absolute -Force }
}
foreach ($entry in $planData.files | Where-Object action -eq 'keep') {
    $source=Get-SafeModelFile $entry.path
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'Retained file changed' }
}
@{status='complete';removed_bytes=$removed;retained_checkpoints=$planData.checkpoint_files_retained;metadata_history=$planData.history_root} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $historyRoot 'cleanup-result.json') -Encoding utf8
Write-Output "PRUNE_COMPLETE retained=$($planData.checkpoint_files_retained) removed_bytes=$removed; no training/data changes"
