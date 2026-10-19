$ErrorActionPreference = "Stop"

python -m src.evaluation.compare_global_periocular_ablation `
    --config configs/global_periocular.yaml

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
