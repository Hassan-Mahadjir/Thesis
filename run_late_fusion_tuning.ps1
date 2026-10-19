$ErrorActionPreference = "Stop"

python -m src.training.train_periocular_metric `
    --config configs/global_periocular_late_fusion.yaml

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "TUNING COMPLETE"
Write-Host "Open reports/training/global_periocular_late_fusion/tuning_summary.json"
Write-Host "Copy best_epoch into final_epochs and oof_fixed_epochs."
