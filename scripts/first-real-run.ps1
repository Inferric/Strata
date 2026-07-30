$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot
Set-Location $RepoDir

if ($env:STRATA_ALLOW_CLOUD -eq "true") {
    throw "The first run is local-only. Set STRATA_ALLOW_CLOUD=false."
}

if (-not $env:MLFLOW_TRACKING_URI) {
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
}
if (-not $env:PREFECT_API_URL) {
    $env:PREFECT_API_URL = "http://localhost:4200/api"
}
if (-not $env:STRATA_NUM_WORKERS) {
    # The Windows spawn probe is reproducible and stable with host-side workers disabled.
    $env:STRATA_NUM_WORKERS = "0"
}

docker compose up -d postgres mlflow prefect api web
docker compose ps
uv run python scripts/wait_for_services.py

if (-not (Test-Path data/raw/otbench/canonical-v1/otbench-mlo-cn2-15m-v1/train_features.csv)) {
    uv run strata-acquire --config configs/data/otbench_mlo.yaml
}
uv run strata-acquire --config configs/data/otbench_mlo.yaml --verify-only

uv run strata-train --config configs/experiments/first_real_mlo.yaml --model mlp --seed 17 --fast-dev-run
uv run strata-train --config configs/experiments/first_real_mlo.yaml --model strata_ot_surface --seed 17 --fast-dev-run
uv run strata-first-flow --config configs/experiments/first_real_mlo.yaml
$gate = Get-Content artifacts/latest/summary.json -Raw | ConvertFrom-Json
if (-not $gate.gate_result.passed) {
    Write-Warning ("Evidence gate remains open: " + ($gate.gate_result.failures -join ", "))
}
$webPort = if ($env:STRATA_WEB_PORT) { $env:STRATA_WEB_PORT } else { "3000" }
Write-Host "First real run complete. Open http://localhost:$webPort and reports/generated/first-real-run/experiment-report.pdf"
