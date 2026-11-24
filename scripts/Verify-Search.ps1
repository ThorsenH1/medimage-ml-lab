param(
    [Parameter(Mandatory)] [string] $Python,
    [Parameter(Mandatory)] [string] $WeightRoot,
    [Parameter(Mandatory)] [string] $SearchRoot,
    [Parameter(Mandatory)] [string] $ChestData,
    [Parameter(Mandatory)] [string] $MammographyData,
    [Parameter(Mandatory)] [string] $SpineData,
    [string] $OriginalArtifacts = 'artifacts'
)

$ErrorActionPreference = 'Stop'
$Python = (Resolve-Path -LiteralPath $Python).Path
$WeightRoot = (Resolve-Path -LiteralPath $WeightRoot).Path
$SearchRoot = (Resolve-Path -LiteralPath $SearchRoot).Path
$ChestData = (Resolve-Path -LiteralPath $ChestData).Path
$MammographyData = (Resolve-Path -LiteralPath $MammographyData).Path
$SpineData = (Resolve-Path -LiteralPath $SpineData).Path

function Invoke-Module {
    & $Python -u -m @args
    if ($LASTEXITCODE -ne 0) { throw "Verification module exited with code $LASTEXITCODE." }
}

Push-Location (Join-Path $PSScriptRoot '..')
try {
    $out = Join-Path $SearchRoot 'inference'
    foreach ($task in @('chest','mammo')) {
        $folder = if ($task -eq 'chest') { 'chest' } else { 'mammo-full' }
        $manifest = Join-Path $OriginalArtifacts "cache/$folder/manifest.csv"
        $chosen = Import-Csv -LiteralPath $manifest | Where-Object { $_.split -eq 'test' -and ($task -eq 'chest' -or ($_.CC -and $_.MLO)) } | Select-Object -First 1
        if (-not $chosen) { throw "No suitable raw image for $task verification." }
        $raw = if ($task -eq 'chest') { @('--image',(Join-Path $ChestData $chosen.path)) } else { @('--cc',(Join-Path $MammographyData $chosen.CC),'--mlo',(Join-Path $MammographyData $chosen.MLO)) }
        Invoke-Module medimage.search_predict radiography --selection (Join-Path $SearchRoot "evaluation/$task/selection.json") --search (Join-Path $SearchRoot "runs/$task") --weights $WeightRoot --original $OriginalArtifacts --output (Join-Path $out "$task.json") @raw
    }
    Invoke-Module medimage.search_predict spine --selection (Join-Path $SearchRoot 'runs/spine/selection.json') --search (Join-Path $SearchRoot 'runs/spine') --weights $WeightRoot --root $SpineData --cache (Join-Path $OriginalArtifacts 'cache/spine-test') --localization (Join-Path $OriginalArtifacts 'spine/localization') --incumbent (Join-Path $OriginalArtifacts 'inference/spine.csv') --output (Join-Path $out 'spine.csv')
    Invoke-Module medimage.search_report --search $SearchRoot --output docs/model-search.md
    Invoke-Module pytest -q
}
finally { Pop-Location }
