param(
    [Parameter(Mandatory)] [string] $Python,
    [Parameter(Mandatory)] [string] $WeightRoot,
    [Parameter(Mandatory)] [string] $SearchRoot,
    [string] $OriginalArtifacts = 'artifacts',
    [ValidateSet('mammo','chest')] [string[]] $Tasks = @('mammo','chest')
)

$ErrorActionPreference = 'Stop'
$Python = (Resolve-Path -LiteralPath $Python).Path
$WeightRoot = (Resolve-Path -LiteralPath $WeightRoot).Path
$SearchRoot = (Resolve-Path -LiteralPath $SearchRoot).Path

Push-Location (Join-Path $PSScriptRoot '..')
try {
    foreach ($task in $Tasks) {
        $cohort = if ($task -eq 'chest') { Join-Path $OriginalArtifacts 'cache/chest' } else { Join-Path $OriginalArtifacts 'cache/mammo-full' }
        $pixels = Join-Path $SearchRoot "pixels/$task"
        $end = (Get-Date).AddMinutes(25)
        while (-not (Test-Path (Join-Path $pixels 'pixels.json'))) {
            if ((Get-Date) -ge $end) { throw "Pixel preparation did not finish for $task." }
            Start-Sleep -Seconds 8
        }
        foreach ($profile in @(@('effb0',160,128),@('r50',224,96),@('effb0',224,112),@('dino',224,96))) {
            $backbone, $resolution, $chestBatch = $profile
            $batch = if ($task -eq 'chest') { $chestBatch } else { 14 }
            $output = Join-Path $SearchRoot "features/$task/$backbone-$resolution"
            & $Python -u -m medimage.search_features extract $task --cohort $cohort --pixels $pixels --weights $WeightRoot --backbone $backbone --resolution $resolution --batch-size $batch --output $output
            if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed for $task/$backbone-$resolution." }
        }
    }
}
finally { Pop-Location }
