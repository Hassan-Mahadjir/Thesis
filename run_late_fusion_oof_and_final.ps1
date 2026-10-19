$ErrorActionPreference = "Stop"
$Config = "configs/global_periocular_late_fusion.yaml"

foreach ($fold in 1..4) {
    python -m src.training.train_periocular_metric `
        --config $Config `
        --oof-fold $fold
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python -m src.evaluation.build_global_periocular_late_fusion_oof `
    --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m src.training.train_periocular_metric `
    --config $Config `
    --final
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m src.evaluation.evaluate_global_periocular_late_fusion `
    --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
