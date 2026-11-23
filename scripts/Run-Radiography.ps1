param(
    [Parameter(Mandatory)] [ValidateSet('chest', 'mammo')] [string] $Task,
    [Parameter(Mandatory)] [string] $Python,
    [Parameter(Mandatory)] [string] $Weights,
    [Parameter(Mandatory)] [string] $DataRoot,
    [string] $ArtifactRoot = 'artifacts'
)

$ErrorActionPreference = 'Stop'
$Python = (Resolve-Path -LiteralPath $Python).Path
$Weights = (Resolve-Path -LiteralPath $Weights).Path
$DataRoot = (Resolve-Path -LiteralPath $DataRoot).Path

function Invoke-PythonModule {
    & $Python -u -m @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE." }
}

Push-Location (Join-Path $PSScriptRoot '..')
try {
    $cache = Join-Path $ArtifactRoot "cache/$Task"
    if ($Task -eq 'mammo') { $cache = Join-Path $ArtifactRoot 'cache/mammo-full' }
    $global = Join-Path $ArtifactRoot "runs/$Task-global"
    $attention = Join-Path $ArtifactRoot "runs/$Task-attention"
    $batch = if ($Task -eq 'chest') { 96 } else { 12 }
    $globalEpochs = if ($Task -eq 'chest') { 5 } else { 12 }
    $attentionEpochs = if ($Task -eq 'chest') { 7 } else { 14 }
    Invoke-PythonModule medimage.radiography_data $Task --root $DataRoot --output $cache
    if ($Task -eq 'chest') {
        Invoke-PythonModule medimage.chest_cohort --cache $cache
    }
    $developmentCache = $cache
    $developmentArgs = @()
    $legacy = Join-Path $ArtifactRoot 'cache/mammo'
    if ($Task -eq 'mammo' -and (Test-Path (Join-Path $legacy 'audit.json')) -and (Test-Path (Join-Path $global 'result.json'))) {
        $original = Get-Content -Raw (Join-Path $legacy 'audit.json') | ConvertFrom-Json
        $run = Get-Content -Raw (Join-Path $global 'result.json') | ConvertFrom-Json
        if ($run.config.manifest -eq $original.manifest_sha256) {
            $developmentCache = $legacy
            $developmentArgs = @('--development-cache', $legacy)
        }
    }
    Invoke-PythonModule medimage.radiography_train $Task --variant global --cache $developmentCache --weights $Weights --output $global --epochs $globalEpochs --batch-size $batch
    Invoke-PythonModule medimage.radiography_train $Task --variant attention --cache $developmentCache --weights $Weights --output $attention --epochs $attentionEpochs --batch-size $batch
    Invoke-PythonModule medimage.radiography_evaluate $Task --cache $cache @developmentArgs --runs $global $attention --output (Join-Path $ArtifactRoot "evaluation/$Task") --batch-size $batch
}
finally { Pop-Location }
