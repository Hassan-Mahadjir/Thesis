$ErrorActionPreference = "Stop"

$Config = "configs/global_periocular.yaml"

$variants = @(
    "global_pair",
    "local_pair",
    "global_local_pair"
)

foreach ($variant in $variants) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "TUNING DEVELOPMENT ABLATION: $variant"
    Write-Host "============================================================"

    python -m src.training.train_global_periocular `
        --config $Config `
        --variant $variant

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "COMPARING DEVELOPMENT RESULTS"
Write-Host "============================================================"

python -m src.evaluation.compare_global_periocular_ablation `
    --config $Config

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Done."
Write-Host "Do NOT run FINAL_TEST yet."
